/**
 * @file algorithm_names.hpp
 * @brief Defines constants for algorithm names used throughout BanditPAM.
 * 
 * This file centralizes algorithm name strings to ensure consistency
 * across the codebase. See GitHub Issue #238.
 */

#ifndef HEADERS_ALGORITHM_NAMES_HPP_
#define HEADERS_ALGORITHM_NAMES_HPP_

#include <string>

namespace km {
namespace AlgorithmNames {
  // Primary algorithms
  const std::string BANDITPAM = "BanditPAM";
  const std::string BANDITPAM_ORIG = "BanditPAM_orig";
  const std::string PAM = "PAM";
  const std::string FASTPAM1 = "FastPAM1";
}  // namespace AlgorithmNames
}  // namespace km

#endif  // HEADERS_ALGORITHM_NAMES_HPP_
