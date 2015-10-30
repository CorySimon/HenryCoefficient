#!/bin/bash

###
### Benchmark CUDA and OpenMP code performance
###
#echo "GPUkernelcalls,time" > GPU_performance.csv
echo "EquivalentGPUkernelcalls,time" > OpenMP_performance.csv

export KMP_AFFINITY=compact
export OMP_NUM_THREADS=72
echo "Running with $OMP_NUM_THREADS OpenMP threads"

# Run codes with varying numbers of Monte Carlo insertions, $n
for n in `seq 16 16 512` ; do
    #echo "Running with $n GPU kernel calls"

    ###
    # GPU code
    ###
    #t=$({ time ./henry_gpu $n >/dev/null; } |& grep real | awk '{print $2}')
    #echo -e "\tCUDA run time: $t"
    #echo "$n,$t" >> GPU_performance.csv  # write results to .csv

    ###
    # OpenMP code
    ###
    t=$({ time ./henry_cpu $n >/dev/null; } |& grep real | awk '{print $2}')
    time ./henry_cpu $n >/dev/null;
    echo -e "\tOpenMP run time: $t"
    echo "$n,$t" >> OpenMP_performance.csv  # write results to .csv

done

#python plot_performance.py
