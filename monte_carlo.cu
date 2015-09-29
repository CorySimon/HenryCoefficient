#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <map>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

// for generating RNG seeds
__host__ __device__
unsigned int hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

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
__host__ __device__ double ComputeBoltzmannFactorAtPoint(double x, double y, double z,
                                                StructureAtom * structureatoms,
                                                int natoms,
                                                double L) {
    // (x, y, z) : Cartesian coords of methane molecule
    // structureatoms : vector storing info on unit cell of crystal structure
    // L : box length
    double E = 0.0;
    
    // loop over atoms in crystal structure
    for (int i = 0; i < natoms; i++) {
        //  Compute distance from (x, y, z) to this atom

        // compute distances in each coordinate
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
        if (dy <= -L / 2.0)
            dy = dy + L;

        // distance
        double rinv = rsqrt(dx*dx + dy*dy + dz*dz);

        // Compute contribution to energy of adsorbate at (x, y, z) due to this atom
        // Lennard-Jones potential (not efficient, but for clarity)
        E += 4.0 * structureatoms[i].epsilon * (pow(structureatoms[i].sigma * rinv, 12) - pow(structureatoms[i].sigma * rinv, 6));
    }
    return exp(-E / (R * T));  // return Boltzmann factor
}

struct estimate_avg_Boltzmann_factor : public thrust::unary_function<unsigned int,double> {
    estimate_avg_Boltzmann_factor(StructureAtom * structureatoms_, int natoms_, double L_) 
        : L(L_), natoms(natoms_), structureatoms(structureatoms_)
        {}
    StructureAtom * structureatoms;
    int natoms;
    double L;
        
    __host__ __device__
    double operator()(unsigned int thread_id) {
        double sum = 0.0;
        unsigned int N = 10000; // samples per thread

        // seed a random number generator for uniform double in [0,1)
        unsigned int seed = hash(thread_id);
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<double> u01(0.0, 1.0);

        // perform N insertions
        for(unsigned int i = 0; i < N; ++i) {
            // Generate random position inside the cubic unit cell of the structure
            double x = L * u01(rng);
            double y = L * u01(rng);
            double z = L * u01(rng);
     
            // Compute Boltzmann factor
            sum += ComputeBoltzmannFactorAtPoint(x, y, z, structureatoms, natoms, L);
        }

        // divide by N for average
        return sum / N;
    }
};

int main(void)
{
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
    thrust::host_vector<StructureAtom> hostStructureatoms;
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

    double L;
    istream >> L;   
    printf("L = %f\n", L);

    // waste line
    getline(materialfile, line);
    
    // get number of atoms
    getline(materialfile, line);
    int natoms;  // number of atoms
    istream.str(line);
    istream.clear();
    istream >> natoms;
    printf("%d atoms\n", natoms);
    
    // waste line
    getline(materialfile, line);

    // read atom coordinates
    for (int i = 0; i < natoms; i++) {
        getline(materialfile, line);
        istream.str(line);
        istream.clear();

        int atomno;
        double xf, yf, zf;  // fractional coordintes
        std::string element;

        istream >> atomno >> element >> xf >> yf >> zf;
        
        StructureAtom thisatom;

        thisatom.x = L * xf;
        thisatom.y = L * yf;
        thisatom.z = L * zf;

        thisatom.epsilon = epsilons[element];
        thisatom.sigma = sigmas[element];

        hostStructureatoms.push_back(thisatom);

//        printf("%d. %s, (%f, %f, %f), eps = %f, sig = %f\n", 
//            atomno, element.c_str(), 
//            structureatoms[i].x, structureatoms[i].y, structureatoms[i].z,
//            structureatoms[i].epsilon,
//            structureatoms[i].sigma);
    }
    
    // copy structure atoms to device
    thrust::device_vector<StructureAtom> deviceStructureatoms = hostStructureatoms;
    // get raw pointer of deviceStructureatoms
    StructureAtom * structureatoms = thrust::raw_pointer_cast(deviceStructureatoms.data());

    // use 30K independent seeds
    int M = 30000;

    double avg_Boltzmann_factor = thrust::transform_reduce(thrust::counting_iterator<int>(0),
                                            thrust::counting_iterator<int>(M),
                                            estimate_avg_Boltzmann_factor(structureatoms, natoms, L),
                                            0.0f,
                                            thrust::plus<double>());
    avg_Boltzmann_factor /= M;
    printf("Henry constant = %e mol/(m3 - Pa)\n", avg_Boltzmann_factor / (R * T));

    return 0;
}
