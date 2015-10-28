#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error %s at %s:%d\n",cudaGetErrorString(x), __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define NUMTHREADS 256  // number of threads per GPU block

// data for atom of crystal structure
//    Unit cell of crystal structure can then be stored 
//    as pointer array of StructureAtom's
struct StructureAtom {
    // Cartesian position, units: A
    double x;
    double y;
    double z;
    // Lennard-Jones epsilon parameter with adsorbate
    double epsilon;  // units: K
    // Lennard-Jones sigma parameter with adsorbate
    double sigma;  // units: A
};

// temperature, Kelvin
const double T = 298.0; 

// Universal gas constant, m3 - Pa / (K - mol)
const double R = 8.314; 

// Compute the Boltzmann factor of methane at point (x, y, z) inside structure
//   Loop over all atoms of unit cell of crystal structure
//   Find nearest image to methane at point (x, y, z) for application of periodic boundary conditions
//   Compute energy contribution due to this atom via the Lennard-Jones potential
__device__ double ComputeBoltzmannFactorAtPoint(double x, double y, double z,
                                                const StructureAtom * __restrict__ structureatoms,
                                                double natoms,
                                                double L) {
    // (x, y, z) : Cartesian coords of methane molecule
    // structureatoms : pointer array storing info on unit cell of crystal structure
    // natoms : number of atoms in crystal structure
    // L : box length
    // returns Boltzmann factor e^{-E/(RT)}
    double E = 0.0;  // energy (K)
    
    // loop over atoms in crystal structure
    for (int i = 0; i < natoms; i++) {
        // Compute distance in each coordinate from (x, y, z) to this structure atom
        double dx = x - structureatoms[i].x;
        double dy = y - structureatoms[i].y;
        double dz = z - structureatoms[i].z;
        
        // apply nearest image convention for periodic boundary conditions
        if (dx > L / 2.0)
            dx = dx - L;
        if (dy > L / 2.0)
            dy = dy - L;
        if (dz > L / 2.0)
            dz = dz - L;
        if (dx <= -L / 2.0)
            dx = dx + L;
        if (dy <= -L / 2.0)
            dy = dy + L;
        if (dz <= -L / 2.0)
            dz = dz + L;

        // compute inverse distance
        double rinv = rsqrt(dx*dx + dy*dy + dz*dz);

        // Compute contribution to energy of adsorbate at (x, y, z) due to this atom
        // Lennard-Jones potential (not efficient, but for clarity)
        E += 4.0 * structureatoms[i].epsilon * (pow(structureatoms[i].sigma * rinv, 12) - pow(structureatoms[i].sigma * rinv, 6));
    }
    return exp(-E / (R * T));  // return Boltzmann factor
}

// Inserts a methane molecule at a random position inside the structure
// Calls function to compute Boltzmann factor at this point
// Stores Boltzmann factor computed at this thread in boltzmannFactors
__global__ void PerformInsertions(curandStateMtgp32 *state, 
                                  double *boltzmannFactors, 
                                  const StructureAtom * __restrict__ structureatoms, 
                                  int natoms, double L) {
    // state : random number generator
    // boltzmannFactors : pointer array in which to store computed Boltzmann factors
    // structureatoms : pointer array storing atoms in unit cell of crystal structure
    // natoms : number of atoms in crystal structure
    // L : box length
    int id = threadIdx.x + blockIdx.x * NUMTHREADS;
        
    // Generate random position inside the cubic unit cell of the structure
    double x = L * curand_uniform_double(&state[blockIdx.x]);
    double y = L * curand_uniform_double(&state[blockIdx.x]);
    double z = L * curand_uniform_double(&state[blockIdx.x]);

    // Compute Boltzmann factor, store in boltzmannFactors
    boltzmannFactors[id] = ComputeBoltzmannFactorAtPoint(x, y, z, structureatoms, natoms, L);
}

int main(int argc, char *argv[]) {
    // take in number of MC insertions as argument
    if (argc != 2) {
        printf("Run as:\n./henry ninsertions\n");
        exit(EXIT_FAILURE);
    }
    int ncycles = atoi(argv[1]);  // Number of Monte Carlo insertions

    //
    // Energetic model for interactions of methane molecule with atoms of framework
    //    pairwise Lennard-Jones potentials
    //
    // Epsilon parameters for Lennard-Jones potential (K)
    std::map<std::string, double> epsilons;
    epsilons["Zn"] = 96.152688;
    epsilons["O"] = 66.884614;
    epsilons["C"] = 88.480032;
    epsilons["H"] = 57.276566;
    
    // Sigma parameters for Lennard-Jones potential (A)
    std::map<std::string, double> sigmas;
    sigmas["Zn"] = 3.095775;
    sigmas["O"] = 3.424075;
    sigmas["C"] = 3.580425;
    sigmas["H"] = 3.150565;

    //
    // Import unit cell of nanoporous material IRMOF-1
    //
    StructureAtom *structureatoms;  // store atoms in pointer array here
    // open crystal structure file
    std::ifstream materialfile("IRMOF-1.cssr");
    if (materialfile.fail()) {
        printf("IRMOF-1.cssr failed to import.\n");
        exit(EXIT_FAILURE);
    }

    // read cubic box dimensions
    std::string line;
    getline(materialfile, line);
    std::istringstream istream(line);

    double L;  // dimension of cube
    istream >> L;
    printf("L = %f\n", L);

    // waste line
    getline(materialfile, line);
    
    // get number of atoms
    getline(materialfile, line);
    int natoms;  // number of atoms in unit cell of IRMOF-1
    istream.str(line);
    istream.clear();
    istream >> natoms;
    printf("%d atoms\n", natoms);
    
    // waste line
    getline(materialfile, line);

    // Allocate space for material atoms and epsilons/sigmas on both host and device
    //   using Unified Memory
    CUDA_CALL(cudaMallocManaged(&structureatoms, natoms * sizeof(StructureAtom)));

    // read atom coordinates
    for (int i = 0; i < natoms; i++) {
        // read atoms from .cssr file
        getline(materialfile, line);
        istream.str(line);
        istream.clear();

        int atomno;
        double xf, yf, zf;  // fractional coordinates
        std::string element;

        istream >> atomno >> element >> xf >> yf >> zf;

        // load structureatoms with Cartesian coordinates of this atom
        structureatoms[i].x = L * xf;
        structureatoms[i].y = L * yf;
        structureatoms[i].z = L * zf;
        
        // store epsilon and sigma Lennard-Jones parameters as well, for ease of computation on device
        structureatoms[i].epsilon = epsilons[element];
        structureatoms[i].sigma = sigmas[element];

//        printf("%d. %s, (%f, %f, %f), eps = %f, sig = %f\n", 
//            atomno, element.c_str(), 
//            structureatoms[i].x, structureatoms[i].y, structureatoms[i].z,
//            structureatoms[i].epsilon,
//            structureatoms[i].sigma);
    }
    
    // Determine number of blocks for high occupancy: 4 per multiprocessor
    int deviceId = -1;
    int nMultiprocessors = 0;
    CUDA_CALL(cudaGetDevice(&deviceId));
    CUDA_CALL(cudaDeviceGetAttribute(&nMultiprocessors, cudaDevAttrMultiProcessorCount, deviceId));
    int nBlocks = nMultiprocessors * 4;   
    
    // calculate number of MC insertions
    int ninsertions = ncycles * nBlocks * NUMTHREADS;
    
    //
    // Allocate space for storing Boltzmann factors computed on each thread using unified memory
    //
    double * boltzmannFactors;
    CUDA_CALL(cudaMallocManaged(&boltzmannFactors, nBlocks * NUMTHREADS * sizeof(double)));
    
    //
    // Set up random number generator on device
    //
    curandStateMtgp32 *devMTGPStates;
    mtgp32_kernel_params *devKernelParams;

    // Allocate space for prng states on device. One per block
    CUDA_CALL(cudaMalloc((void **) &devMTGPStates, nBlocks * sizeof(curandStateMtgp32))); 
    
    // Setup MTGP prng states
    // Allocate space for MTGP kernel parameters
    CUDA_CALL(cudaMalloc((void**) &devKernelParams, sizeof(mtgp32_kernel_params))); 
    
    // Reformat from predefined parameter sets to kernel format,
    // and copy kernel parameters to device memory
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams)); 
    
    // Initialize one state per thread block
    CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, nBlocks, 1234)); 
    // State setup is complete
    
    //
    //  Compute the Henry coefficient
    //  KH = < e^{-E/(kB * T)} > / (R * T)
    //  Brackets denote average over space
    //

    double KH = 0.0;  // will be Henry coefficient
    for (int cycle = 0; cycle < ncycles; cycle++) {
        //  Perform Monte Carlo insertions in parallel on the GPU
        PerformInsertions<<<nBlocks, NUMTHREADS>>>(devMTGPStates, boltzmannFactors, structureatoms, natoms, L);
        cudaDeviceSynchronize();

        // Compute Henry coefficient from the sampled Boltzmann factors (will be average Boltzmann factor divided by RT)
        for(int i = 0; i < nBlocks * NUMTHREADS; i++)
            KH += boltzmannFactors[i];
    }
    KH = KH / ninsertions;  // --> average Boltzmann factor
    // at this point KH = < e^{-E/(kB/T)} >
    KH = KH / (R * T);  // divide by RT
    printf("Used %d blocks with %d thread each\n", nBlocks, NUMTHREADS);
    printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
    printf("Number of actual insertions: %d\n", ninsertions);
    printf("Number of times we called the GPU kernel: %d\n", ncycles);
    
    // Clean-up
    CUDA_CALL(cudaFree(devMTGPStates));
    CUDA_CALL(cudaFree(structureatoms));
    CUDA_CALL(cudaFree(boltzmannFactors));
    return EXIT_SUCCESS;
}
