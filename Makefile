CC = nvcc
MPCC = nvcc
OPENMP = 
CFLAGS = -O3
NVCCFLAGS = -O3 -arch sm_30
LIBS = -lm

all: henry henry_serial monte_carlo

henry: henry.o
	$(CC) -o $@ $(NVCCLIBS) -L/usr/local/cuda/lib64 henry.o 

henry.o: henry.cu 
	$(CC) -c $(NVCCFLAGS) henry.cu

henry_serial: henry_serial.o
	g++ -o $@ henry_serial.o -fopenmp

henry_serial.o: henry_serial.cc
	g++ -c henry_serial.cc -std=c++11 -fopenmp

monte_carlo: monte_carlo.o
	$(CC) -o $@ $(NVCCLIBS) -L/usr/local/cuda/lib64 henry.o 

monte_carlo.o: monte_carlo.cu
	$(CC) -c $(NVCCFLAGS) monte_carlo.cu

clean: 
	rm *.o 
