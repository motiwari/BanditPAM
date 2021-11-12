#ifndef HEADERS_LOG_HELPER_HPP_
#define HEADERS_LOG_HELPER_HPP_

#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string>

namespace km {
    /**
     *  \brief Logging class for structured KMedoids logs.
     *
     *  LogHelper class. Assists the KMedoids class in structured logging.
     */
struct LogHelper {
        std::ofstream hlogFile; // Output stream that writes the KMedoids log

        std::vector<double> comp_exact_build; // Number of computations in build step
        std::vector<double> comp_exact_swap; // Number of computations in swap step

        std::vector<double> loss_build; // Loss after each iteration of build step
        std::vector<double> loss_swap; // Loss after each iteration of swap step

        std::vector<double> p_build; // Precision for each iteration of build step
        std::vector<double> p_swap; // Precision for each iteration of swap step

        std::vector<std::string> sigma_build; // Distributions for each iteration of build step
        std::vector<std::string> sigma_swap; // Distributions for each iteration of swap step

        /*! \brief Opens the log file.
        *
        *  Opens the log file.
        *
        *  @param input_filename Filename that log will be saved as.
        */
        void init(const std::string& input_filename = "KMedoidsLogfile") {
            hlogFile.open(input_filename);
        }

        /*! \brief Closes the log file.
        *
        *  Closes the log file.
        */
        void close() {
            hlogFile.close();
        }

        /*! \brief Writes a vector out for a given key
        *
        *  Writes a vector out for a given key
        *
        *  @param key Key for json-ified output structure
        *  @param vec Vector to be iterated across when writing line
        */
        void writeSummaryLine(const std::string& key, arma::rowvec vec) {
            hlogFile << key << ':';
            for (size_t i = 0; i < vec.n_cols; i++) {
                if (i == (vec.n_cols - 1)) {
                    hlogFile << vec(i) << '\n';
                } else {
                    hlogFile << vec(i) << ',';
                }
            }
        }

        /*! \brief Writes a logstring component
        *
        *  Writes a logstring line for the writeProfile function.
        *
        *  @param key Key for json-ified output structure
        *  @param vec Vector to be iterated across when writing logstring line
        */
        template <typename T>
        void writeLogStringLine(const std::string& key, std::vector<T> vec) {
            hlogFile << "\t\t:" << key << '\n';
            for (size_t i = 0; i < vec.size(); i++) {
                hlogFile << "\t\t\t\t" << i << ": " << vec.at(i) << '\n';
            }
        }

        /*! \brief Writes formatted summary log of a KMedoids run
        *
        *  Writes summary statistics of a KMedoids run. Statistics include medoids
        *  after the build step, medoids after the swap step, number of swap steps,
        *  the final loss, and logstrings of the number of points that had distance
        *  computations, loss, precision, and uncertainty for each iteration of
        *  both the build and swap steps.
        *
        *  @param b_medoids Medoids after the build step.
        *  @param f_medoids Medoids after the swap step (final medoids).
        *  @param steps Number of swap steps.
        *  @param loss Final loss of the KMedoids object.
        */
        void writeProfile(const arma::rowvec& b_medoids, const arma::rowvec& f_medoids, size_t steps, double loss) {
            writeSummaryLine("Built", b_medoids);
            writeSummaryLine("Swapped", f_medoids);
            hlogFile << "Num Swaps: " << steps << '\n';
            hlogFile << "Final Loss: " << loss << '\n';

            hlogFile << "Build Logstring:" << '\n';
            writeLogStringLine("compute_exactly", comp_exact_build);
            writeLogStringLine("loss", loss_build);
            writeLogStringLine("p", p_build);
            writeLogStringLine("sigma", sigma_build);

            hlogFile << "Swap Logstring:" << '\n';
            writeLogStringLine("compute_exactly", comp_exact_swap);
            writeLogStringLine("loss", loss_swap);
            writeLogStringLine("p", p_swap);
            writeLogStringLine("sigma", sigma_swap);
        }
};
} // namespace km
#endif // HEADERS_LOG_HELPER_HPP_
