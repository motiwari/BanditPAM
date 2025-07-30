/**
 * @file loss_fn_python.cpp
 * @date 2024-01-01
 *
 * Defines the function getLossFn in KMedoidsWrapper class
 * which converts LossType enum to string for Python interface.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
std::string km::KMedoidsWrapper::getLossFn() const {
  LossType lossType = KMedoids::getLossFn();
  switch (lossType) {
    case LossType::MANHATTAN:
      return "manhattan";
    case LossType::COS:
      return "cos";
    case LossType::COSINE:
      return "cosine";
    case LossType::INF:
      return "L-infinity";
    case LossType::EUCLIDEAN:
      return "euclidean";
    case LossType::LP_NORM:
      return "L" + std::to_string(getLp());
    case LossType::UNKNOWN:
    default:
      throw std::invalid_argument("Error: Loss Function Undefined!");
  }
}
}  // namespace km 