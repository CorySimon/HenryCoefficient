#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <map>
#include<random>

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
double ComputeBoltzmannFactorAtPoint(double x, double y, double z,
                                     const StructureAtom * restrict structureatoms,
                                     int natoms, double L)
{
    // (x, y, z) : Cartesian coords of methane molecule
    // structureatoms : pointer array storing info on unit cell of crystal structure
    // natoms : number of atoms in crystal structure
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
        double r = sqrt(dx*dx + dy*dy + dz*dz);

        // Compute contribution to energy of adsorbate at (x, y, z) due to this atom
        // Lennard-Jones potential (not efficient, but for clarity)
        E += 4.0 * structureatoms[i].epsilon * (pow(structureatoms[i].sigma / r, 12) - pow(structureatoms[i].sigma / r, 6));
    }
    return exp(-E / (R * T));  // return Boltzmann factor
}

// Inserts a methane molecule at a random position inside the structure
// Calls function to compute Boltzmann factor at this point
// Stores Boltzmann factor computed at this thread in deviceBoltzmannFactors
int main(int argc, char *argv[]) {
    // take in number of MC insertions as argument
    if (argc != 2) {
        printf("Run as:\n./henry ninsertions\nwhere ninsertions = Number of MC insertions / (256 * 64) to correspond to CUDA code");
        exit(EXIT_FAILURE);
    }
    const int ninsertions = atoi(argv[1]) * 256 * 64;  // Number of Monte Carlo insertions

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
    StructureAtom * structureatoms;  // store data in pointer array here
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

    // Allocate space for material atoms and epsilons/sigmas
    structureatoms = (StructureAtom *) malloc(natoms * sizeof(StructureAtom));

    // read atom coordinates
    for (int i = 0; i < natoms; i++) {
        getline(materialfile, line);
        istream.str(line);
        istream.clear();

        int atomno;
        double xf, yf, zf;  // fractional coordintes
        std::string element;

        istream >> atomno >> element >> xf >> yf >> zf;
        // load structureatoms
        structureatoms[i].x = L * xf;
        structureatoms[i].y = L * yf;
        structureatoms[i].z = L * zf;

        structureatoms[i].epsilon = epsilons[element];
        structureatoms[i].sigma = sigmas[element];

//        printf("%d. %s, (%f, %f, %f), eps = %f, sig = %f\n", 
//            atomno, element.c_str(), 
//            structureatoms[i].x, structureatoms[i].y, structureatoms[i].z,
//            structureatoms[i].epsilon,
//            structureatoms[i].sigma);
    }
    
    //
    // Set up random number generator
    //
    const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    //
    //  Compute the Henry coefficient in parallel
    //  KH = < e^{-E/(kB * T)} > / (R * T)
    //  Brackets denote average over space
    //
    double KH = 0.0;  // will be Henry coefficient
    #pragma omp parallel default(none) firstprivate(L,natoms,seed) shared(KH,structureatoms)
    {
        std::default_random_engine generator(seed);  // default
        std::uniform_real_distribution<double> uniform01(0.0, 1.0); // uniformly distributed real no in [0,1]

        #pragma omp for reduction (+:KH)
        for (int i = 0; i < ninsertions; i++) {
            // generate random position in structure
            double x = L * uniform01(generator);
            double y = L * uniform01(generator);
            double z = L * uniform01(generator);
            // compute Boltzmann factor
            KH += ComputeBoltzmannFactorAtPoint(x, y, z, structureatoms, natoms, L);
        }
    }
    // KH = < e^{-E/(kB/T)} > / (RT)
    KH = KH / (ninsertions * R * T);
    printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
    printf("Number of insertions: %d\n", ninsertions);
    
    return EXIT_SUCCESS;
}
