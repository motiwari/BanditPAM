/**
 * @file fasterpam.cpp
 * @date 2021-07-25
 *
 * Contains a C++ implementation of the FasterPAM algorithm.
 * The original FasterPAM papers are:
 * 1) Erich Schubert and Peter J. Rousseeuw: Fast and Eager k-Medoids Clustering:
 *  O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
 * 2) Erich Schubert and Peter J. Rousseeuw: Faster k-Medoids Clustering:
 *  Improving the PAM, CLARA, and CLARANS Algorithms
 */

#include "fasterpam.hpp"

#include <tuple>
#include <armadillo>
#include <vector>
#include <cassert>
#include <random>
#include <time.h> // TODO(@Adarsh321123): remove this
#include <sys/time.h> // TODO(@Adarsh321123): remove this

double get_wall_time(){ // TODO(@Adarsh321123): remove this
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    //  Handle error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

namespace km {
DistancePair::DistancePair(
  size_t i,
  float d) {
  this->i = i;
  this->d = d;
}

DistancePair DistancePair::empty() {
  return DistancePair(std::numeric_limits<size_t>::max(), 0.0);
}

Rec::Rec() : near(0, 0.0), seco(0, 0.0) {}

Rec::Rec(
  size_t i1,
  float d1,
  size_t i2,
  float d2)
    : near(i1, d1), seco(i2, d2)
{
}

Rec Rec::empty() {
  return Rec(
      DistancePair::empty().i,
      DistancePair::empty().d,
      DistancePair::empty().i,
      DistancePair::empty().d
  );
}

void FasterPAM::fitFasterPAM(
  const arma::fmat& inputData,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
  data = arma::trans(inputData);
  // FasterPAM requires a distance matrix
  // so, we will now compute one
  arma::fmat mat(data.n_cols, data.n_cols);
  for (int m = 0; m < data.n_cols; m++) {
    for (int n = m; n < data.n_cols; n++) {
      // TODO(@Adarsh321123): add lots of comments
      // TODO(@Adarsh321123): ensure that output from main is correct
      float dist = (this->*lossFn)(data, m, n);
      mat(m, n) = dist;
      mat(n, m) = dist;
    }
  }
  double wall0 = get_wall_time(); // TODO(@Adarsh321123): remove this
  arma::urowvec medoidIndices = randomInitialization(data.n_cols);
  steps = 0;
  medoidIndicesBuild = medoidIndices;
  size_t n = mat.n_rows;
  arma::urowvec assignments(n, arma::fill::zeros);
  // TODO(@Adarsh321123): make it so no need for things to be returned (i.e. change things like averageLoss)
  // TODO(@Adarsh321123): pass assi by reference so no need to pass back?
  // TODO(@Adarsh321123): use pointer and references in parameters and arguments like other algorithms
  std::tuple<float, arma::urowvec, size_t, size_t> paramsFasterPAM = FasterPAM::swapFasterPAM(data, mat, medoidIndices, assignments);
  double wall1 = get_wall_time(); // TODO(@Adarsh321123): remove this
  float loss = std::get<0>(paramsFasterPAM) / data.n_cols;
  assignments = std::get<1>(paramsFasterPAM);
  size_t iter = std::get<2>(paramsFasterPAM);
  size_t swaps = std::get<3>(paramsFasterPAM);
  std::cout << "FasterPAM final loss: " << loss << std::endl;
  std::cout << "FasterPAM swaps performed: " << swaps << std::endl;
  std::cout << "Medoids: " << std::endl;
  for (size_t i = 0; i < medoidIndices.size(); i++) {
    std::cout << i << " -> " << medoidIndices[i] << std::endl;
  }
  std::cout << "Wall Clock Time: " << wall1 - wall0 << "\n"; // TODO(@Adarsh321123): remove this

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
  steps = iter;
}

arma::urowvec FasterPAM::randomInitialization(
  size_t n) {
  // from https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
  const size_t rangeFrom = 0;
  const size_t rangeTo = n-1;
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_int_distribution<size_t> distr(rangeFrom, rangeTo);
  arma::urowvec res(nMedoids);
  for (size_t i = 0; i < nMedoids; i++) {
    res[i] = distr(generator);
  }
  return res;
}

std::tuple<float, std::vector<Rec>> FasterPAM::initialAssignment(
  const arma::fmat& mat,
  arma::urowvec medoidIndices) {
  size_t n = mat.n_rows;
  assert(("Dissimilarity matrix is not square",
          mat.n_rows == mat.n_cols));
  assert(("N is too large", n <= std::numeric_limits<size_t>::max()));
  assert(("invalid N", nMedoids > 0 && nMedoids < std::numeric_limits<size_t>::max()));
  assert(("k must be at most N", nMedoids <= n));
  std::vector<Rec> distanceData(mat.n_rows);
  for (size_t i = 0; i < mat.n_rows; i++) {
    distanceData[i] = Rec::empty();
  }

  size_t firstCenter = medoidIndices[0];
  float loss = 0.0;
  for (size_t i = 0; i < distanceData.size(); i++) {
    Rec& cur = distanceData[i];
    cur = Rec(0, mat(i, firstCenter), std::numeric_limits<size_t>::max(), 0.0);
    for (size_t m = 1; m < medoidIndices.size(); m++) {
      size_t me = medoidIndices[m];
      float d = mat(i, me);
      if (d < cur.near.d || i == me) {
        cur.seco = cur.near;
        cur.near = DistancePair(m, d);
      } else if (cur.seco.i == std::numeric_limits<size_t>::max() || d < cur.seco.d) {
        cur.seco = DistancePair(m, d);
      }
    }
    loss += cur.near.d;
  }
  return {loss, distanceData};
}

void FasterPAM::debugAssertAssignment(
  const arma::fmat& mat,
  arma::urowvec medoidIndices,
  std::vector<Rec>& distanceData) {
  for (size_t o = 0; o < mat.n_rows; o++) {
    assert(("primary assignment inconsistent", mat(o, medoidIndices[distanceData[o].near.i]) == distanceData[o].near.d));
    assert(("secondary assignment inconsistent", mat(o, medoidIndices[distanceData[o].seco.i]) == distanceData[o].seco.d));
    assert(("nearest is farther than second nearest", distanceData[o].near.d <= distanceData[o].seco.d));
  }
}

std::tuple<bool, float> FasterPAM::chooseMedoidWithinPartition(
  const arma::fmat& mat,
  arma::urowvec assignments,
  arma::urowvec& medoidIndices,
  size_t m) {
  size_t first = medoidIndices[m];
  size_t best = first;
  float sumb = 0.0;
  for (size_t i = 0; i < assignments.size(); i++) {
    size_t a = assignments[i];
    if (first != i && a == m) {
      sumb += mat(first, i);
    }
  }
  for (size_t j = 0; j < assignments.size(); j++) {
    size_t aj = assignments[j];
    if (j != first && aj == m) {
      float sumj = 0.0;
      for (size_t i = 0; i < assignments.size(); i++) {
        size_t ai = assignments[i];
        if (i != j && ai == m) {
          sumj += mat(j, i);
        }
      }
      if (sumj < sumb) {
        best = j;
        sumb = sumj;
      }
    }
  }
  medoidIndices[m] = best;
  return {best != first, sumb};
}

void FasterPAM::updateRemovalLoss(
  std::vector<Rec>& distanceData,
  arma::frowvec& loss) {
  loss.fill(0.0);
  for (Rec rec : distanceData) {
    loss[rec.near.i] += rec.seco.d - rec.near.d;
  }
}

std::tuple<float, size_t> FasterPAM::findBestSwap(
  const arma::fmat& mat,
  arma::frowvec& removalLoss,
  std::vector<Rec>& distanceData,
  size_t j) {
  arma::frowvec ploss = removalLoss;
  float acc = 0.0;
  for (size_t o = 0; o < distanceData.size(); o++) {
    Rec reco = distanceData[o];
    float djo = mat(j, o);
    if (djo < reco.near.d) {
      acc += djo - reco.near.d;
      ploss[reco.near.i] += reco.near.d - reco.seco.d;
    } else if (djo < reco.seco.d) {
      ploss[reco.near.i] += djo - reco.seco.d;
    }
  }
  auto it = std::min_element(std::begin(ploss), std::end(ploss));
  size_t b = std::distance(std::begin(ploss), it);
  float bloss = *it;
  return {bloss + acc, b};
}

DistancePair FasterPAM::updateSecondNearest(
  const arma::fmat& mat,
  arma::urowvec medoidIndices,
  size_t n,
  size_t b,
  size_t o,
  float djo) {
  DistancePair s = DistancePair(b, djo);
  for (size_t i = 0; i < medoidIndices.size(); i++) {
    size_t mi = medoidIndices[i];
    if (i == n || i == b) {
      continue;
    }
    float d = mat(o, mi);
    if (d < s.d) {
      s = DistancePair(i, d);
    }
  }
  return s;
}

float FasterPAM::doSwap(
  const arma::fmat& mat,
  arma::urowvec& medoidIndices,
  std::vector<Rec>& distanceData,
  size_t b,
  size_t j) {
  size_t n = mat.n_rows;
  assert(("invalid medoid number",
          b < medoidIndices.size()));
  assert(("invalid object number",
          j < n));
  medoidIndices[b] = j;
  float loss = 0.0;
  for (size_t o = 0; o < distanceData.size(); o++) {
    Rec& reco = distanceData[o];
    if (o == j) {
      if (reco.near.i != b) {
        reco.seco = reco.near;
      }
      reco.near = DistancePair(b, 0);
      continue;
    }
    float djo = mat(j, o);
    if (reco.near.i == b) {
      if (djo < reco.seco.d) {
        reco.near = DistancePair(b, djo);
      } else {
        reco.near = reco.seco;
        reco.seco = updateSecondNearest(mat, medoidIndices, reco.near.i, b, o, djo);
      }
    } else {
      if (djo < reco.near.d) {
        reco.seco = reco.near;
        reco.near = DistancePair(b, djo);
      } else if (djo < reco.seco.d) {
        reco.seco = DistancePair(b, djo);
      } else if (reco.seco.i == b) {
        reco.seco = updateSecondNearest(mat, medoidIndices, reco.near.i, b, o, djo);
      }
    }
    loss += reco.near.d;
  }
  return loss;
}

std::tuple<float, arma::urowvec, size_t, size_t> FasterPAM::swapFasterPAM(
  const arma::fmat &data,
  const arma::fmat& mat,
  arma::urowvec& medoidIndices,
  arma::urowvec assignments) {
  size_t n = mat.n_rows; // TODO(@Adarsh321123): remove duplication with that of fit function
  if (nMedoids == 1) {
    std::tuple<bool, float> paramsMedoid = chooseMedoidWithinPartition(mat, assignments, medoidIndices, 0);
    bool swapped = std::get<0>(paramsMedoid);
    float loss = std::get<1>(paramsMedoid);
    return {loss, assignments, 1, (swapped) ? 1 : 0};
  }
  std::tuple<float, std::vector<Rec>> paramsAssi = initialAssignment(mat, medoidIndices);
  float loss = std::get<0>(paramsAssi);
  std::vector<Rec> distanceData = std::get<1>(paramsAssi);
  debugAssertAssignment(mat, medoidIndices, distanceData);
  arma::frowvec removalLoss(nMedoids, arma::fill::zeros);
  updateRemovalLoss(distanceData, removalLoss);
  size_t lastSwap = n;
  size_t nSwaps = 0;
  size_t iter = 0;
  while (iter < maxIter) {
    iter++;
    size_t swapsBefore = nSwaps;
    for (size_t j = 0; j < n; j++) {
      if (j == lastSwap) {
        break;
      }
      if (j == medoidIndices[distanceData[j].near.i]) {
        continue;
      }
      std::tuple<float, size_t> paramsSwap = findBestSwap(mat, removalLoss, distanceData, j);
      float change = std::get<0>(paramsSwap);
      size_t b = std::get<1>(paramsSwap);
      if (change >= 0) {
        continue;
      }
      nSwaps++;
      lastSwap = j;
      float newLoss = doSwap(mat, medoidIndices, distanceData, b, j);
      if (newLoss >= loss) {
        break;
      }
      loss = newLoss;
      updateRemovalLoss(distanceData, removalLoss);
    }
    if (nSwaps == swapsBefore) {
      break;
    }
  }

  for (size_t i = 0; i < distanceData.size(); i++) {
    const auto& x = distanceData[i];
    assignments[i] = static_cast<size_t>(x.near.i);
  }
  return { loss, assignments, iter, nSwaps };
}
}  // namespace km
