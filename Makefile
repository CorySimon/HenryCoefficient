CC        := icc
CXX       := icpc
MPCC 	  := nvcc
CFLAGS    := -O3 -std=c99 -qopenmp
CXXFLAGS  := -O3 -std=c++11 -qopenmp
NVCCFLAGS := -O3 -arch sm_30
CXXLIBS   := -lm
NVCCLIBS  := -L/usr/local/cuda/lib64

all: henry_cpu # henry_gpu

henry_gpu: henry_gpu.o
	$(CC) $(NVCCFLAGS)$< $(NVCCLIBS) -o $@

henry_gpu.o: henry_gpu.cu
	$(CC) $(NVCCFLAGS) -c $< -o $@

henry_cpu: henry_cpu.o
	$(CXX) $(CXXFLAGS) $< -o $@

henry_cpu.o: henry_cpu.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm *.o
