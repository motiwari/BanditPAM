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
DistancePair::DistancePair(size_t i, float d) {
  this->i = i;
  this->d = d;
}

DistancePair DistancePair::empty() {
  return DistancePair(std::numeric_limits<size_t>::max(), 0.0);
}

Rec::Rec() : near(0, 0.0), seco(0, 0.0) {}

Rec::Rec(size_t i1, float d1, size_t i2, float d2)
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
  // FasterPAM requires a distance matrix
  // so, we will now compute one
  arma::fmat mat(inputData.n_rows, inputData.n_rows);
  for (int m = 0; m < inputData.n_rows; m++) {
    for (int n = m; n < inputData.n_rows; n++) {
      // TODO(@Adarsh321123): implement and test multiple distances
      // TODO(@Adarsh321123): add lots of comments
      float dist = arma::norm(inputData.row(m) - inputData.row(n), 2);
      mat(m, n) = dist;
      mat(n, m) = dist;
    }
  }
  double wall0 = get_wall_time(); // TODO(@Adarsh321123): remove this
  // TODO(@Adarsh321123): refactor names (i.e. meds -> medoidIndices)
  // TODO(@Adarsh321123): don't use vector when possible, (i.e. use arma::urowvec for medoidIndices)
  std::vector<size_t> meds = randomInitialization(inputData.n_rows, nMedoids);
  steps = 0;
//  medoidIndicesBuild = meds; // TODO: come back to this once using arma::urowvec
  size_t n = mat.n_rows;
  std::vector<size_t> assi(n);
  for (size_t i = 0; i < n; i++) {
    assi[i] = 0;
  }
  // TODO(@Adarsh321123): make it so no need for things to be returned (i.e. change things like averageLoss)
  // TODO(@Adarsh321123): pass assi by reference so no need to pass back?
  std::tuple<float, std::vector<size_t>, size_t, size_t> paramsFasterPAM = FasterPAM::swapFasterPAM(inputData, mat, meds, assi);
  double wall1 = get_wall_time(); // TODO(@Adarsh321123): remove this
  float loss = std::get<0>(paramsFasterPAM) / inputData.n_rows;
  assi = std::get<1>(paramsFasterPAM);
  size_t iter = std::get<2>(paramsFasterPAM);
  size_t swaps = std::get<3>(paramsFasterPAM);
  std::cout << "FasterPAM final loss: " << loss << std::endl;
  std::cout << "FasterPAM swaps performed: " << swaps << std::endl;
  std::cout << "Medoids: " << std::endl;
  for (size_t i = 0; i < meds.size(); i++) {
    std::cout << i << " -> " << meds[i] << std::endl;
  }
  std::cout << "Wall Clock Time: " << wall1 - wall0 << "\n"; // TODO(@Adarsh321123): remove this

//  medoidIndicesFinal = meds; // TODO: come back to this once using arma::urowvec
//  labels = assi; // TODO: come back to this once using arma::urowvec
}

std::vector<size_t> FasterPAM::randomInitialization(
  size_t n,
  size_t k) {
  // from https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
  const size_t rangeFrom = 0;
  const size_t rangeTo = n-1;
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_int_distribution<size_t> distr(rangeFrom, rangeTo);
  std::vector<size_t> res(k);
  for (size_t i = 0; i < k; i++) {
    res[i] = distr(generator);
  }
  return res;
}

std::tuple<float, std::vector<Rec>> FasterPAM::initialAssignment(
  const arma::fmat& mat,
  std::vector<size_t> med) {
  size_t n = mat.n_rows;
  size_t k = med.size();
  assert(("Dissimilarity matrix is not square",
          mat.n_rows == mat.n_cols));
  assert(("N is too large", n <= std::numeric_limits<size_t>::max()));
  assert(("invalid N", k > 0 && k < std::numeric_limits<size_t>::max()));
  assert(("k must be at most N", k <= n));
  std::vector<Rec> data(mat.n_rows);
  for (size_t i = 0; i < mat.n_rows; i++) {
    data[i] = Rec::empty();
  }

  size_t firstCenter = med[0];
  float loss = 0.0;
  for (size_t i = 0; i < data.size(); i++) {
    Rec& cur = data[i];
    cur = Rec(0, mat(i, firstCenter), std::numeric_limits<size_t>::max(), 0.0);
    for (size_t m = 1; m < med.size(); m++) {
      size_t me = med[m];
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
  return {loss, data};
}

void FasterPAM::debugAssertAssignment(
  const arma::fmat& mat,
  std::vector<size_t> med,
  std::vector<Rec>& data) {
  for (size_t o = 0; o < mat.n_rows; o++) {
    assert(("primary assignment inconsistent", mat(o, med[data[o].near.i]) == data[o].near.d));
    assert(("secondary assignment inconsistent", mat(o, med[data[o].seco.i]) == data[o].seco.d));
    assert(("nearest is farther than second nearest", data[o].near.d <= data[o].seco.d));
  }
}

std::tuple<bool, float> FasterPAM::chooseMedoidWithinPartition(
  const arma::fmat& mat,
  std::vector<size_t> assi,
  std::vector<size_t>& med,
  size_t m) {
  size_t first = med[m];
  size_t best = first;
  float sumb = 0.0;
  for (size_t i = 0; i < assi.size(); i++) {
    size_t a = assi[i];
    if (first != i && a == m) {
      sumb += mat(first, i);
    }
  }
  for (size_t j = 0; j < assi.size(); j++) {
    size_t aj = assi[j];
    if (j != first && aj == m) {
      float sumj = 0.0;
      for (size_t i = 0; i < assi.size(); i++) {
        size_t ai = assi[i];
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
  med[m] = best;
  return {best != first, sumb};
}

void FasterPAM::updateRemovalLoss(std::vector<Rec>& data, std::vector<float>& loss) {
  for (size_t i = 0; i < loss.size(); i++) {
    loss[i] = 0.0;
  }
  for (Rec rec : data) {
    loss[rec.near.i] += rec.seco.d - rec.near.d;
  }
}

std::tuple<float, size_t> FasterPAM::findBestSwap(
  const arma::fmat& mat,
  std::vector<float>& removalLoss,
  std::vector<Rec>& data,
  size_t j) {
  std::vector<float> ploss = removalLoss;
  float acc = 0.0;
  for (size_t o = 0; o < data.size(); o++) {
    Rec reco = data[o];
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
  std::vector<size_t> med,
  size_t n,
  size_t b,
  size_t o,
  float djo) {
  DistancePair s = DistancePair(b, djo);
  for (size_t i = 0; i < med.size(); i++) {
    size_t mi = med[i];
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
  std::vector<size_t>& med,
  std::vector<Rec>& data,
  size_t b,
  size_t j) {
  size_t n = mat.n_rows;
  assert(("invalid medoid number",
          b < med.size()));
  assert(("invalid object number",
          j < n));
  med[b] = j;
  float loss = 0.0;
  for (size_t o = 0; o < data.size(); o++) {
    Rec& reco = data[o];
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
        reco.seco = updateSecondNearest(mat, med, reco.near.i, b, o, djo);
      }
    } else {
      if (djo < reco.near.d) {
        reco.seco = reco.near;
        reco.near = DistancePair(b, djo);
      } else if (djo < reco.seco.d) {
        reco.seco = DistancePair(b, djo);
      } else if (reco.seco.i == b) {
        reco.seco = updateSecondNearest(mat, med, reco.near.i, b, o, djo);
      }
    }
    loss += reco.near.d;
  }
  return loss;
}

std::tuple<float, std::vector<size_t>, size_t, size_t> FasterPAM::swapFasterPAM(
  const arma::fmat &inputData,
  const arma::fmat& mat,
  std::vector<size_t>& med,
  std::vector<size_t> assi) {
  size_t n = mat.n_rows; // TODO(@Adarsh321123): remove duplication with that of fit function
  // TODO: don't need to find k here since nMedoids is known, same for other vars when applicable
  size_t k = med.size();
  if (k == 1) {
    std::tuple<bool, float> paramsMedoid = chooseMedoidWithinPartition(mat, assi, med, 0);
    bool swapped = std::get<0>(paramsMedoid);
    float loss = std::get<1>(paramsMedoid);
    return {loss, assi, 1, (swapped) ? 1 : 0};
  }
  std::tuple<float, std::vector<Rec>> paramsAssi = initialAssignment(mat, med);
  float loss = std::get<0>(paramsAssi);
  std::vector<Rec> data = std::get<1>(paramsAssi);
  debugAssertAssignment(mat, med, data);
  std::vector<float> removalLoss(k);
  for (size_t i = 0; i < k; i++) {
    removalLoss[i] = 0.0;
  }
  updateRemovalLoss(data, removalLoss);
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
      if (j == med[data[j].near.i]) {
        continue;
      }
      std::tuple<float, size_t> paramsSwap = findBestSwap(mat, removalLoss, data, j);
      float change = std::get<0>(paramsSwap);
      size_t b = std::get<1>(paramsSwap);
      if (change >= 0) {
        continue;
      }
      nSwaps++;
      lastSwap = j;
      float newLoss = doSwap(mat, med, data, b, j);
      if (newLoss >= loss) {
        break;
      }
      loss = newLoss;
      updateRemovalLoss(data, removalLoss);
    }
    if (nSwaps == swapsBefore) {
      break;
    }
  }
  assi.clear();
  for (const auto& x : data) {
    assi.push_back(static_cast<size_t>(x.near.i));
  }
  return { loss, assi, iter, nSwaps };
}
}  // namespace km
