# ZK-SPoW GPU Benchmarks
# Requires: NVIDIA GPU + CUDA toolkit (nvcc)
#
# Targets:
#   make bench          # Poseidon2 throughput benchmark (Pure PoW vs Merkle vs Symbiotic)
#   make stark          # Full GPU STARK prover + symbiotic benchmark
#   make all            # build both
#
# Options:
#   ARCH=sm_89          # Ada Lovelace (RTX 4070/4090)
#   ARCH=sm_80          # Ampere (A100/RTX 3090) [default]
#   ARCH=sm_90          # Hopper (H100)
#
# Run:
#   ./bench [difficulty] [log_array_size] [seconds]
#   ./stark [log_trace] [difficulty] [seconds]
#   ./stark sweep [difficulty] [seconds]

NVCC ?= nvcc
ARCH ?= sm_80

NVFLAGS = -O3 -arch=$(ARCH) --std=c++17

all: bench stark

bench: bench.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

stark: stark.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

run-bench: bench
	./bench 24 22 5

run-stark: stark
	./stark 16 24 5

run-sweep: stark
	./stark sweep 24 3

clean:
	rm -f bench stark

.PHONY: all run-bench run-stark run-sweep clean
