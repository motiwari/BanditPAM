/**
 * @file kmedoids_algorithm.cpp
 * @date 2020-06-10
 *
 * Contains the primary C++ implementation of the BanditPAM code.
 */

#include <omp.h>
#include <armadillo>
#include <unordered_map>
#include <regex>
#include <cassert>

#include "kmedoids_algorithm.hpp"
#include "fastpam1.hpp"
#include "pam.hpp"
#include "banditpam.hpp"
#include "banditpam_orig.hpp"

namespace km {

// NOTE: The order of arguments in this constructor must match that of the
// arguments in kmedoids_pywrapper.cpp, otherwise undefined behavior can
// result (variables being initialized with others' values)
KMedoids::KMedoids(size_t nMedoids, const std::string &algorithm,
                   size_t maxIter, size_t buildConfidence,
                   size_t swapConfidence, bool useCache, bool usePerm,
                   size_t cacheWidth, bool parallelize, size_t seed)
  : nMedoids(nMedoids),
    algorithm(algorithm),
    maxIter(maxIter),
    buildConfidence(buildConfidence),
    swapConfidence(swapConfidence),
    useCache(useCache),
    usePerm(usePerm),
    cacheWidth(cacheWidth),
    parallelize(parallelize),
    seed(seed) {
  KMedoids::checkAlgorithm(algorithm);
  // Though we initialize seed from the given parameter,
  // we need to call setSeed to pass it to arma
  KMedoids::setSeed(seed);
}

KMedoids::~KMedoids() {}

void KMedoids::fit(
  const arma::fmat &inputData, const std::string &loss,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
  numMiscDistanceComputations = 0;
  numBuildDistanceComputations = 0;
  numSwapDistanceComputations = 0;
  numCacheWrites = 0;
  numCacheHits = 0;
  numCacheMisses = 0;

  if (distMat) {  // User has provided a distance matrix
    if (distMat.value().get().n_cols != distMat.value().get().n_rows) {
      // TODO(@motiwari): Change this to an assertion
      //  that is properly raised
      throw std::invalid_argument("Malformed distance matrix provided");
    }
    useDistMat = true;
  } else {
    // In case the user is running a new problem
    // without a distance matrix
    // after running a distance matrix problem
    useDistMat = false;
  }

  if (inputData.n_rows == 0) {
    // TODO(@motiwari): Change this to an assertion
    //  that is properly raised
    throw std::invalid_argument("Dataset is empty");
  }

  assert(("Number of medoids should be less than the number of data points",
          nMedoids < inputData.n_rows));

  batchSize = fmin(inputData.n_rows, batchSize);

  try {
    KMedoids::setLossFn(loss);
    if (algorithm == "PAM") {
      static_cast<PAM *>(this)->fitPAM(inputData, distMat);
    } else if (algorithm == "BanditPAM") {
      static_cast<BanditPAM *>(this)->fitBanditPAM(inputData, distMat);
    } else if (algorithm == "BanditPAM_orig") {
      static_cast<BanditPAM_orig *>(this)->fitBanditPAM_orig(inputData,
                                                             distMat);
    } else if (algorithm == "FastPAM1") {
      static_cast<FastPAM1 *>(this)->fitFastPAM1(inputData, distMat);
    }
  } catch (std::invalid_argument &e) {
    std::cout << e.what() << std::endl;
    std::cout << "Error: Clustering did not run." << std::endl;
    throw e;
  }
}

arma::urowvec KMedoids::getMedoidsBuild() const { return medoidIndicesBuild; }

arma::urowvec KMedoids::getMedoidsFinal() const { return medoidIndicesFinal; }

arma::urowvec KMedoids::getLabels() const { return labels; }

size_t KMedoids::getSteps() const { return steps; }

size_t KMedoids::getNMedoids() const { return nMedoids; }

void KMedoids::setNMedoids(size_t newNMedoids) { nMedoids = newNMedoids; }

std::string KMedoids::getAlgorithm() const { return algorithm; }

void KMedoids::setAlgorithm(const std::string &newAlgorithm) {
  algorithm = newAlgorithm;
  KMedoids::checkAlgorithm(algorithm);
}

size_t KMedoids::getMaxIter() const { return maxIter; }

void KMedoids::setMaxIter(size_t newMaxIter) { maxIter = newMaxIter; }

size_t KMedoids::getBuildConfidence() const { return buildConfidence; }

void KMedoids::setBuildConfidence(size_t newBuildConfidence) {
  if (algorithm != "BanditPAM" && algorithm != "BanditPAM_orig") {
    // TODO(@motiwari): Better error type
    throw "Cannot set buildConfidence when not using BanditPAM";
  }
  buildConfidence = newBuildConfidence;
}

size_t KMedoids::getSwapConfidence() const { return swapConfidence; }

void KMedoids::setSwapConfidence(size_t newSwapConfidence) {
  if (algorithm != "BanditPAM" && algorithm != "BanditPAM_orig") {
    // TODO(@motiwari): Better error type
    throw "Cannot set buildConfidence when not using BanditPAM";
  }
  swapConfidence = newSwapConfidence;
}

void KMedoids::setSeed(size_t newSeed) {
  seed = newSeed;
  arma::arma_rng::set_seed(seed);
}

size_t KMedoids::getSeed() const { return seed; }

bool KMedoids::getUseCache() const { return useCache; }

void KMedoids::setUseCache(bool newUseCache) {
  // TODO(@motiwari): Throw an error if not using BanditPAM
  useCache = newUseCache;
}

bool KMedoids::getUsePerm() const { return usePerm; }

void KMedoids::setUsePerm(bool newUsePerm) {
  // TODO(@motiwari): Throw an error if not using BanditPAM
  usePerm = newUsePerm;
}

size_t KMedoids::getCacheWidth() const {
  // TODO(@motiwari): Throw an error if not using BanditPAM
  return cacheWidth;
}

void KMedoids::setCacheWidth(size_t newCacheWidth) {
  // TODO(@motiwari): Throw an error if not using BanditPAM
  cacheWidth = newCacheWidth;
}

bool KMedoids::getParallelize() const { return parallelize; }

// TODO(@motiwari): Change this to const bool newParallelize
void KMedoids::setParallelize(bool newParallelize) {
  parallelize = newParallelize;
}

size_t KMedoids::getDistanceComputations(const bool includeMisc) const {
  if (includeMisc) {
    return numMiscDistanceComputations + numBuildDistanceComputations +
           numSwapDistanceComputations;
  } else {
    return numBuildDistanceComputations + numSwapDistanceComputations;
  }
}

size_t KMedoids::getMiscDistanceComputations() const {
  return numMiscDistanceComputations;
}

size_t KMedoids::getBuildDistanceComputations() const {
  return numBuildDistanceComputations;
}

size_t KMedoids::getSwapDistanceComputations() const {
  return numSwapDistanceComputations;
}

size_t KMedoids::getCacheWrites() const { return numCacheWrites; }

size_t KMedoids::getCacheHits() const { return numCacheHits; }

size_t KMedoids::getCacheMisses() const { return numCacheMisses; }

size_t KMedoids::getTotalSwapTime() const { return totalSwapTime; }

float KMedoids::getTimePerSwap() const { return totalSwapTime / steps; }

void KMedoids::setLossFn(std::string loss) {
  // TODO(@motiwari): On setting this, clear the
  //  cache and the average loss,
  // assignments, medoids, etc.
  std::for_each(loss.begin(), loss.end(), [](char &c) {
    c = ::tolower(c);  // TODO(@motiwari): Put something before ::
  });
  switch (getLossType(loss)) {
    case LossType::MANHATTAN:
      lossFn = &KMedoids::manhattan;
      break;
    case LossType::COS:
    case LossType::COSINE:
      lossFn = &KMedoids::cos;
      break;
    case LossType::INF:
      lossFn = &KMedoids::LINF;
      break;
    case LossType::EUCLIDEAN:
      lossFn = &KMedoids::LP;
      lp = 2;
      break;
    case LossType::LP_NORM:
      lossFn = &KMedoids::LP;
      lp = stoi(loss.substr(1));
      break;
    case LossType::UNKNOWN:
    default:
      throw std::invalid_argument("Error: unrecognized loss function");
  }
}

std::string KMedoids::getLossFn() const {
  // TODO(@motiwari): make the strings constants
  if (lossFn == &KMedoids::manhattan) {
    return "manhattan";
  } else if (lossFn == &KMedoids::cos) {
    return "cosine";
  } else if (lossFn == &KMedoids::LINF) {
    return "L-infinity";
  } else if (lossFn == &KMedoids::LP) {
    return "L" + std::to_string(lp);
  } else {
    throw std::invalid_argument("Error: Loss Function Undefined!");
  }
}

void KMedoids::calcBestDistancesSwap(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::urowvec *medoidIndices, arma::frowvec *bestDistances,
  arma::frowvec *secondBestDistances, arma::urowvec *assignments,
  const bool swapPerformed) {
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < data.n_cols; i++) {
    float best = std::numeric_limits<float>::infinity();
    float second = std::numeric_limits<float>::infinity();
    for (size_t k = 0; k < medoidIndices->n_cols; k++) {
      float cost = KMedoids::cachedLoss(data, distMat, i, (*medoidIndices)(k),
                                        AlgorithmStep::MISC);
      if (cost < best) {
        (*assignments)(i) = k;
        second = best;
        best = cost;
      } else if (cost < second) {
        second = cost;
      }
    }
    (*bestDistances)(i) = best;
    (*secondBestDistances)(i) = second;
  }

  if (!swapPerformed) {
    // We have converged; update the final loss
    averageLoss = arma::accu(*bestDistances) / data.n_cols;
  }
}

float KMedoids::calcLoss(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::urowvec *medoidIndices) {
  float total = 0;
// TODO(@motiwari): is this parallel loop accumulating properly?
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < data.n_cols; i++) {
    float cost = std::numeric_limits<float>::infinity();
    for (size_t k = 0; k < nMedoids; k++) {
      float currCost = KMedoids::cachedLoss(
        data, distMat, i, (*medoidIndices)(k), AlgorithmStep::MISC);
      if (currCost < cost) {
        cost = currCost;
      }
    }
    total += cost;
  }

  // Returns average distance
  return total / data.n_cols;
}

float KMedoids::cachedLoss(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const size_t i, const size_t j, AlgorithmStep step,
  const bool useCacheFunctionOverride) {
  if (step == AlgorithmStep::MISC) {
    numMiscDistanceComputations++;
  } else if (step == AlgorithmStep::BUILD) {
    numBuildDistanceComputations++;
  } else if (step == AlgorithmStep::SWAP) {
    numSwapDistanceComputations++;
  } else {
    throw std::invalid_argument(
      "Unknown AlgorithmStep in KMedoids::cachedLoss");
  }

  if (this->useDistMat) {
    return distMat.value().get().at(i, j);
  }

  if (!useCache) {
    return (this->*lossFn)(data, i, j);
  }

  // TODO(@motiwari): Should infer n and m from the size of the cache
  size_t n = data.n_cols;
  size_t m = fmin(n, cacheWidth);

  // test this is one of the early points in the permutation
  if (reindex.find(j) != reindex.end()) {
    // TODO(@motiwari): Potential race condition with shearing?
    // T1 begins to write to cache and then T2
    // access in the middle of write?
    if (cache[(m * i) + reindex[j]] == -1) {
      // cache miss! calculate the distance and cache it.
      numCacheMisses++;
      numCacheWrites++;
      cache[(m * i) + reindex[j]] = (this->*lossFn)(data, i, j);
    } else {
      numCacheHits++;
    }
    return cache[m * i + reindex[j]];
  }

  numCacheMisses++;
  return (this->*lossFn)(data, i, j);
}

void KMedoids::checkAlgorithm(const std::string &algorithm) const {
  if ((algorithm != "BanditPAM") && (algorithm != "BanditPAM_orig") &&
      (algorithm != "PAM") && (algorithm != "FastPAM1")) {
    // TODO(@motiwari): Better error type
    throw "unrecognized algorithm";
  }
}

float KMedoids::getAverageLoss() const { return averageLoss; }

float KMedoids::getBuildLoss() const { return buildLoss; }

float KMedoids::LP(const arma::fmat &data, const size_t i,
                   const size_t j) const {
  return arma::norm(data.col(i) - data.col(j), lp);
}

float KMedoids::LINF(const arma::fmat &data, const size_t i,
                     const size_t j) const {
  return arma::max(arma::abs(data.col(i) - data.col(j)));
}

float KMedoids::cos(const arma::fmat &data, const size_t i,
                    const size_t j) const {
  return 1 - (arma::dot(data.col(i), data.col(j)) /
              (arma::norm(data.col(i)) * arma::norm(data.col(j))));
}

float KMedoids::clippedCos(const arma::fmat &data, const size_t i,
                           const size_t j) const {
  // Calculate the cosine distance
  float cos = (arma::dot(data.col(i), data.col(j)) /
               (arma::norm(data.col(i)) * arma::norm(data.col(j))));

  if (cos < 0.3) {
    return 1;  // Cosine distance is too large, so we consider the similarity as
               // zero
  } else {
    return 1 - cos;  // Cosine distance is within the threshold, return it
  }
}

float KMedoids::manhattan(const arma::fmat &data, const size_t i,
                          const size_t j) const {
  return arma::accu(arma::abs(data.col(i) - data.col(j)));
}

float KMedoids::pearson(const arma::fmat &data, const size_t i,
                        const size_t j) const {
  const arma::fvec &xi = data.col(i);
  const arma::fvec &xj = data.col(j);
  float mean_i = arma::mean(xi);
  float mean_j = arma::mean(xj);
  float numerator = arma::dot(xi - mean_i, xj - mean_j);
  float denominator = std::sqrt(arma::dot(xi - mean_i, xi - mean_i) *
                                arma::dot(xj - mean_j, xj - mean_j));
  return 1 - numerator / denominator;
}

arma::fvec KMedoids::rank(const arma::fvec &vec) const {
  arma::uvec sortedIndices = arma::sort_index(vec);
  arma::fvec ranks(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    ranks(sortedIndices(i)) = i + 1;
  }
  return ranks;
}

float KMedoids::spearman(const arma::fmat &data, const size_t i,
                         const size_t j) const {
  arma::fvec rank_i = rank(data.col(i));
  arma::fvec rank_j = rank(data.col(j));
  return pearson(arma::join_rows(rank_i, rank_j), 0, 1);
}
}  // namespace km
