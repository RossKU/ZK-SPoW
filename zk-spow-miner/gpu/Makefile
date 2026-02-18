# ZK-SPoW GPU Benchmark
# Requires: NVIDIA GPU + CUDA toolkit (nvcc)
#
# Usage:
#   make              # build for sm_80 (Ampere, e.g. A100/RTX 3090)
#   make ARCH=sm_89   # build for Ada Lovelace (RTX 4090)
#   make ARCH=sm_90   # build for Hopper (H100)
#   make run           # build and run with defaults
#   make run ARGS="24 22 10"  # difficulty=24, array=2^22, 10sec/phase

NVCC ?= nvcc
ARCH ?= sm_80
ARGS ?= 24 22 5

NVFLAGS = -O3 -arch=$(ARCH) --std=c++17

bench: bench.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

run: bench
	./bench $(ARGS)

clean:
	rm -f bench

.PHONY: run clean
