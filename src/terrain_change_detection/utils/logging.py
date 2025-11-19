"""
Logging Utilities

This module sets up logging for the project and includes helpers to
redirect noisy library stdout/stderr to our logger at a chosen level.
"""

import logging
import sys
from pathlib import Path
import os
import tempfile
from typing import Optional
from contextlib import contextmanager, redirect_stdout, redirect_stderr


def setup_logger(name: str,
                 level: int = logging.INFO,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: logging.INFO)
        log_file: Optional log file path. If provided, logs will be written to this file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers
    if logger.handlers:
        return logger

    # Set the logging level
    logger.setLevel(level)

    # Create formatters: simpler for console, detailed for file
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(processName)s[%(process)d] | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        # Create parent directories if they don't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class _StreamToLogger:
    """
    File-like stream object that redirects writes to a logger.

    - Buffers partial lines until newline.
    - If a pattern is provided, only lines containing the pattern are logged; others
      are optionally passed through to the original stream.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG,
                 pattern: Optional[str] = None, passthrough_stream=None):
        self.logger = logger
        self.level = level
        self.pattern = pattern
        self.passthrough = passthrough_stream
        self._buffer = ""

    def write(self, msg: str) -> int:
        if not isinstance(msg, str):
            msg = str(msg)
        self._buffer += msg
        # Process full lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(msg)

    def flush(self) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        # Flush passthrough if present
        try:
            if self.passthrough and hasattr(self.passthrough, "flush"):
                self.passthrough.flush()
        except Exception:
            pass

    def _handle_line(self, line: str) -> None:
        text = line.rstrip()
        if not text:
            return
        try:
            if self.pattern is None or (self.pattern in text):
                self.logger.log(self.level, text)
            elif self.passthrough and hasattr(self.passthrough, "write"):
                self.passthrough.write(line + "\n")
        except Exception:
            # Never crash due to logging redirection issues
            pass


@contextmanager
def redirect_stdout_stderr_to_logger(logger: logging.Logger,
                                     level: int = logging.DEBUG,
                                     pattern: Optional[str] = None,
                                     passthrough_other: bool = True):
    """
    Context manager that redirects stdout and stderr to a logger.

    Args:
        logger: Target logger
        level: Logging level to use (default: DEBUG)
        pattern: If provided, only lines containing the substring are logged
        passthrough_other: If True and pattern is set, non-matching lines are
            written to the original stream; otherwise they are suppressed.
    """
    orig_out = sys.stdout
    orig_err = sys.stderr
    out_stream = _StreamToLogger(logger, level=level, pattern=pattern,
                                 passthrough_stream=(orig_out if (pattern and passthrough_other) else None))
    err_stream = _StreamToLogger(logger, level=level, pattern=pattern,
                                 passthrough_stream=(orig_err if (pattern and passthrough_other) else None))
    with redirect_stdout(out_stream), redirect_stderr(err_stream):
        yield


@contextmanager
def capture_c_streams_to_logger(
    logger: logging.Logger,
    level: int = logging.DEBUG,
    include_patterns: Optional[list[str]] = None,
):
    """
    Capture C-level stdout/stderr (fd 1/2) writes and log matching lines.

    Works by dup/redirecting OS-level file descriptors to temp files, then reading
    them back after the block. This catches prints from C/C++ extensions that bypass
    Python's sys.stdout/sys.stderr.

    On Windows and POSIX.

    Args:
        logger: Target logger
        level: Log level for captured lines
        include_patterns: If provided, only lines containing any of these substrings
            are logged; others are dropped.
    """
    # Duplicate original fds
    orig_out_fd = os.dup(1)
    orig_err_fd = os.dup(2)

    # Temp files for capture
    out_tmp = tempfile.TemporaryFile(mode="w+b")
    err_tmp = tempfile.TemporaryFile(mode="w+b")

    try:
        # Suppress Python logging during fd redirection to avoid writes to invalid handles (Windows safety)
        prev_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        # Redirect stdout/stderr at fd level
        os.dup2(out_tmp.fileno(), 1)
        os.dup2(err_tmp.fileno(), 2)

        yield
    finally:
        try:
            # Restore fds first
            os.dup2(orig_out_fd, 1)
            os.dup2(orig_err_fd, 2)
        except Exception:
            pass
        finally:
            try:
                os.close(orig_out_fd)
            except Exception:
                pass
            try:
                os.close(orig_err_fd)
            except Exception:
                pass

        # Re-enable logging
        try:
            logging.disable(prev_disable)
        except Exception:
            logging.disable(logging.NOTSET)

        def _drain_and_log(tmpfile):
            try:
                tmpfile.flush()
                tmpfile.seek(0)
                data = tmpfile.read()
                text = data.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if not line:
                        continue
                    if include_patterns is None or any(p in line for p in include_patterns):
                        try:
                            logger.log(level, line)
                        except Exception:
                            pass
            except Exception:
                pass
            finally:
                try:
                    tmpfile.close()
                except Exception:
                    pass

        _drain_and_log(out_tmp)
        _drain_and_log(err_tmp)
