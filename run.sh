#!/bin/bash

###
### CUDA code performance
###
echo "NUMBLOCKS,time" > GPU_performance.csv

# Run GPU code with varying numbers of blocks
for b in 1 2 4 5 8 10 16 # 20 25 32 40 50 64
do
    # compile with this many blocks
    grep -rl "#define NUMBLOCKS" henry.cu | xargs sed -i "s/#define NUMBLOCKS.*$/#define NUMBLOCKS $b/g" henry.cu
    make

    echo "Running with $b blocks"

    # run and time
    t=$({ time ./henry >/dev/null; } |& grep real | awk '{print $2}')
    echo -e "\tRun time: $t"

    # write results to .csv
    echo "$b,$t" >> GPU_performance.csv

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
