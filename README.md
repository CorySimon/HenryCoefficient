# CUDA and C++/OpenMP code for computing the Henry coefficient in IRMOF-1

These codes were used to generate the results in the Parallel For All Blog.

:honeybee: The crystal structure of metal-organic framework IRMOF-1 is stored in `IRMOF-1.cssr`. There are 424 atoms in this unit cell, which is a cube of dimension 25.832 Angstroms. The second column in the .cssr is the atom name; the following three columns give fractional coordinates of these atoms in the a, b, and c crystal lattice directions, respectively.

:honeybee: The 'henry.cu' code is the CUDA code for GPUs.

:honeybee: The 'henry_serial.cc' code is the C++ code parallelized using OpenMP. See the:

    #pragma parallel for

that parallelizes the loop.
