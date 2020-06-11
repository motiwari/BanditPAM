/**
 * @file main.cpp
 * @date 2020-06-10
 * 
 * This file defines a command line program that can be used
 * to run the BanditPAM KMedoids algorithim.
 * 
 * Usage
 * ./pam [path/to/input] [number of clusters] -a
 */

#include "kmedoids_ucb.hpp"
#include <unistd.h>
#include <armadillo>
#include <chrono>
#include <fstream>

int main(int argc, char *argv[]) {
    std::string input_name;
    int k;
    bool print_assignments = false;
    int opt; 
    while((opt = getopt(argc, argv, "f:k:a")) != -1)  
    {  
        switch(opt)  
        {
            case 'a':  
                print_assignments = true; 
                break;  
            case 'f':  
                input_name = optarg;
                break;  
            case 'k':  
                k = std::stoi(optarg);
                break;  
            case ':':  
                printf("option needs a value\n");  
                return 1;  
            case '?':  
                printf("unknown option: %c\n", optopt); 
                return 1;
        }  
    }  
    arma::mat data;
    data.load(input_name);
    data = arma::trans(data);

    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;
    std::cout << "Read in " 
        << n 
        << " data points of dimension " 
        << d
        << std::endl;

    arma::urowvec assignments(n);
    arma::urowvec medoid_indices(k);

    KMediods kmed;
    auto start = std::chrono::steady_clock::now();
    kmed.cluster(data, k, assignments, medoid_indices);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Took "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count()
         << " milliseconds" << std::endl;

    // printing out assigments
    if (print_assignments) {
        for (size_t i = 0; i < n; i++) {
            std::cout << assignments(i) << ", ";
        }
        std::cout << std::endl;
    }

}