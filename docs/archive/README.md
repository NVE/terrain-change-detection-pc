# Archived Documentation

**Date Archived**: November 9, 2025  
**Reason**: Outdated planning documents superseded by new strategy

## What's Here

This folder contains planning documents created before the out-of-core tiling infrastructure was completed. These documents assumed a different starting point and have been superseded by the current implementation plans.

### Archived Files

1. **REVISED_PLAN.md**
   - Early GPU acceleration plan
   - Assumed tiling was partially complete
   - Focused on enhancing existing tiling with GPU/JIT

2. **PERFORMANCE_OPTIMIZATION_PLAN.md**
   - Comprehensive optimization strategy
   - Created before tiling system was complete
   - 12-week implementation roadmap with GPU/parallel/out-of-core together

3. **GPU_IMPLEMENTATION_GUIDE.md**
   - Practical GPU implementation guide
   - CUDA setup, hardware abstraction layer
   - Still contains useful reference material but premature for current phase

4. **OPTIMIZATION_SUMMARY.md**
   - Executive summary of optimization analysis
   - Performance bottleneck identification
   - Technology stack recommendations

5. **ROADMAP.md**
   - Original roadmap with parallel/GPU combined approach
   - 18-task breakdown
   - Phase structure different from current plan

6. **QUICK_REFERENCE.md**
   - One-page cheat sheet for original plan
   - Quick reference for GPU setup and APIs

## Why Archived

These documents were created with incomplete information about the out-of-core tiling infrastructure. Key issues:

1. **Wrong Assumptions**: Documents assumed tiling was partial or missing
2. **Wrong Prioritization**: Attempted parallel + GPU + out-of-core simultaneously
3. **Complexity**: Too much at once without clear phases
4. **Reality Check**: After merging `feat/outofcore-tiling`, we discovered:
   - Tiling is **production-ready** and well-designed
   - Sequential tile processing is the clear bottleneck
   - Natural two-phase approach: parallelize tiles first, then GPU acceleration

## Current Documentation

The strategy has been reassessed and new plans created:

### Active Planning Documents

1. **PARALLELIZATION_PLAN.md** (Phase 1)
   - Focus: CPU multi-process parallelization of tile processing
   - Duration: 4 weeks
   - Target: 6-12x speedup
   - Leverages existing tiling infrastructure

2. **GPU_ACCELERATION_PLAN.md** (Phase 2)
   - Focus: GPU acceleration of operations within tiles
   - Duration: 5 weeks (after Phase 1)
   - Target: Additional 5-15x speedup (30-50x total)
   - CuPy/cuML/Numba stack

3. **ROADMAP.md**
   - High-level timeline and strategy
   - Two-phase approach with clear milestones
   - Performance targets and success criteria
   - Risk management

### Supporting Documentation

- **CHANGELOG.md**: Implementation history and decisions
- **CONFIGURATION_GUIDE.md**: YAML configuration reference
- **ALGORITHMS.md**: Algorithm descriptions (DoD, C2C, M3C2)

## Useful Content to Extract

While these documents are outdated as plans, they contain useful reference material:

### From GPU_IMPLEMENTATION_GUIDE.md:
- CUDA installation instructions (Windows/Linux)
- GPU hardware detection code patterns
- CuPy/cuML API examples
- Common issues and solutions

### From PERFORMANCE_OPTIMIZATION_PLAN.md:
- Bottleneck analysis methodology
- Technology stack comparisons
- Hardware recommendations
- Benchmark strategies

### From OPTIMIZATION_SUMMARY.md:
- Performance profiling techniques
- Expected speedup calculations
- Risk identification

**Recommendation**: When implementing Phase 2 (GPU acceleration), reference these archived documents for technical details, but follow the new GPU_ACCELERATION_PLAN.md for strategy and architecture.

## Historical Context

**Timeline**:
- **Early Nov 2025**: Initial optimization planning (these documents)
- **Nov 5-9, 2025**: `feat/outofcore-tiling` completed and matured
- **Nov 9, 2025**: Merged tiling branch, reassessed strategy, archived old plans
- **Nov 9, 2025+**: Current two-phase implementation plan

**Lesson Learned**: Always ensure foundation is solid before planning optimizations. The out-of-core tiling completion changed the optimization landscape entirely—for the better.

---

*These documents are preserved for reference but should not be used as current implementation guides.*
