#!/bin/bash


###
### CUDA code performance
###
 # echo "NUMBLOCKS,time" > GPU_performance.csv
 # 
 # # Run GPU code with varying numbers of blocks
 # for b in `seq 1 63`
 # do
 #     echo "Running with $b blocks"
 # 
 #     # compile with this many blocks
 #     grep -rl "#define NUMBLOCKS" henry.cu | xargs sed -i "s/#define NUMBLOCKS.*$/#define NUMBLOCKS $b/g" henry.cu
 #     make
 # 
 #     # run and time
 #     t=$({ time ./henry >/dev/null; } |& grep real | awk '{print $2}')
 #     echo "$b,$t" >> GPU_performance.csv
 # 
 # done

###
### OpenMP code performance
###
echo "OMP_NUM_THREADS,time" > OpenMP_performance.csv
for c in `seq 1 24`
do
    export OMP_NUM_THREADS=$c
    echo "Running with $OMP_NUM_THREADS OpenMP threads"
    time ./henry_serial
    t=$({ time ./henry_serial >/dev/null; } |& grep real | awk '{print $2}')
    echo "$c,$t" >> OpenMP_performance.csv
done
