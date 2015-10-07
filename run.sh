#!/bin/bash

###
### CUDA code performance
###
echo "NUMTHREADS,time" > GPU_performance.csv

# Run GPU code with varying numbers of threads per block
for nt in 32 64 96 128 160 192 224 256
do
    # compile with this many threads per block
    grep -rl "#define NUMTHREADS" henry.cu | xargs sed -i "s/#define NUMTHREADS.*$/#define NUMTHREADS $nt/g" henry.cu
    make

    echo "Running with $nt threads/block"

    # run and time
    t=$({ time ./henry >/dev/null; } |& grep real | awk '{print $2}')
    echo -e "\tRun time: $t"

    # write results to .csv
    echo "$nt,$t" >> GPU_performance.csv

done

###
### OpenMP code performance
###
echo "OMP_NUM_THREADS,time" > OpenMP_performance.csv
for c in `seq 1 8`
do
    export OMP_NUM_THREADS=$c
    echo "Running with $OMP_NUM_THREADS OpenMP threads"
    time ./henry_serial
    t=$({ time ./henry_serial >/dev/null; } |& grep real | awk '{print $2}')
    echo "$c,$t" >> OpenMP_performance.csv
done

python plot_performance.py
