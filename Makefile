CC = nvcc
MPCC = nvcc
OPENMP = 
CFLAGS = -O3
NVCCFLAGS = -O3 -arch sm_35
LIBS = -lm

all: henry henry_serial

henry: henry.o
	$(CC) -o $@ $(NVCCLIBS) -L/usr/local/cuda/lib64 henry.o 

henry.o: henry.cu 
	$(CC) -c $(NVCCFLAGS) henry.cu

henry_serial: henry_serial.o
	g++ -o $@ henry_serial.o -fopenmp

henry_serial.o: henry_serial.cc
	g++ -c henry_serial.cc -std=c++11 -fopenmp

clean: 
	rm *.o 
