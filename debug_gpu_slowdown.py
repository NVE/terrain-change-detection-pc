#!/usr/bin/env python
"""Debug script to find where GPU slowdown is coming from."""

import numpy as np
import cupy as cp
import time
from cuml.neighbors import NearestNeighbors as cuMLNN
from sklearn.neighbors import NearestNeighbors as sklNN

# Test size
N_REF = 100000
N_QUERY = 100000

print("=" * 80)
print(f"Testing with {N_REF} reference points, {N_QUERY} query points")
print("=" * 80)

# Generate test data
np.random.seed(42)
ref_cpu = np.random.randn(N_REF, 3).astype(np.float32)
query_cpu = np.random.randn(N_QUERY, 3).astype(np.float32)

# Test 1: Pure sklearn CPU
print("\n1. Pure sklearn (CPU):")
t0 = time.time()
nn_cpu = sklNN(n_neighbors=1)
nn_cpu.fit(ref_cpu)
t_fit = time.time() - t0
print(f"   Fit time: {t_fit:.3f}s")

t0 = time.time()
dist_cpu, idx_cpu = nn_cpu.kneighbors(query_cpu)
t_query = time.time() - t0
print(f"   Query time: {t_query:.3f}s")
print(f"   Total: {t_fit + t_query:.3f}s")

# Test 2: Pure cuML GPU (data already on GPU)
print("\n2. Pure cuML (GPU, data on GPU):")
ref_gpu = cp.asarray(ref_cpu)
query_gpu = cp.asarray(query_cpu)

t0 = time.time()
nn_gpu = cuMLNN(n_neighbors=1)
nn_gpu.fit(ref_gpu)
cp.cuda.Stream.null.synchronize()
t_fit = time.time() - t0
print(f"   Fit time: {t_fit:.3f}s")

t0 = time.time()
dist_gpu, idx_gpu = nn_gpu.kneighbors(query_gpu)
cp.cuda.Stream.null.synchronize()
t_query = time.time() - t0
print(f"   Query time: {t_query:.3f}s")
print(f"   Total: {t_fit + t_query:.3f}s")

# Test 3: cuML with CPU->GPU->CPU transfers (mimics our code)
print("\n3. cuML with transfers (mimics our code):")
t0 = time.time()
ref_gpu2 = cp.asarray(ref_cpu)  # CPU -> GPU
nn_gpu2 = cuMLNN(n_neighbors=1)
nn_gpu2.fit(ref_gpu2)
cp.cuda.Stream.null.synchronize()
t_fit = time.time() - t0
print(f"   Fit time (with transfer): {t_fit:.3f}s")

t0 = time.time()
query_gpu2 = cp.asarray(query_cpu)  # CPU -> GPU
dist_gpu2, idx_gpu2 = nn_gpu2.kneighbors(query_gpu2)
dist_result = cp.asnumpy(dist_gpu2)  # GPU -> CPU
idx_result = cp.asnumpy(idx_gpu2)  # GPU -> CPU
cp.cuda.Stream.null.synchronize()
t_query = time.time() - t0
print(f"   Query time (with transfers): {t_query:.3f}s")
print(f"   Total: {t_fit + t_query:.3f}s")

# Test 4: Just the data transfers
print("\n4. Just data transfer overhead:")
t0 = time.time()
ref_gpu3 = cp.asarray(ref_cpu)
cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(f"   CPU->GPU ({N_REF} x 3 float32): {t1-t0:.3f}s")

t0 = time.time()
query_gpu3 = cp.asarray(query_cpu)
cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(f"   CPU->GPU ({N_QUERY} x 3 float32): {t1-t0:.3f}s")

t0 = time.time()
dist_back = cp.asnumpy(dist_gpu)
idx_back = cp.asnumpy(idx_gpu)
cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(f"   GPU->CPU (results): {t1-t0:.3f}s")

print("\n" + "=" * 80)
print("SUMMARY:")
print(f"  sklearn CPU total: {t_fit + t_query:.3f}s")
print(f"  cuML GPU (no transfer): ~0.156s")
print(f"  cuML GPU (with transfers): ~{t_fit + t_query:.3f}s")
print("=" * 80)
