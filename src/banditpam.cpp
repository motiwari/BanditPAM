/**
 * @file banditpam.cpp
 * @date 2021-07-25
 *
 * This file contains the primary C++ implementation of the BanditPAM code.
 *
 */

#include "banditpam.hpp"

#include <armadillo>
#include <unordered_map>
#include <cmath>

namespace km {
/**
 * \brief Runs BanditPAM algorithm.
 *
 * Run the BanditPAM algorithm to identify a dataset's medoids.
 *
 * @param input_data Input data to find the medoids of
 */
void BanditPAM::fit_bpam(const arma::mat& input_data) {
  data = input_data;
  data = arma::trans(data);

  if (this->use_cache_p) {
    size_t n = data.n_cols;
    size_t m = fmin(n, ceil(log10(data.n_cols) * cache_multiplier));
    cache = new float[n * m];

    #pragma omp parallel for
    for (size_t idx = 0; idx < m*n; idx++) {
      cache[idx] = -1;
    }

    permutation = arma::randperm(n);
    permutation_idx = 0;
    reindex = {};
    // TODO(@motiwari): Can we parallelize this?
    for (size_t counter = 0; counter < m; counter++) {
        reindex[permutation[counter]] = counter;
    }
  }

  arma::mat medoids_mat(data.n_rows, n_medoids);
  arma::urowvec medoid_indices(n_medoids);
  // runs build step
  BanditPAM::build(data, &medoid_indices, &medoids_mat);
  steps = 0;

  medoid_indices_build = medoid_indices;
  arma::urowvec assignments(data.n_cols);
  // runs swap step
  BanditPAM::swap(data, &medoid_indices, &medoids_mat, &assignments);
  medoid_indices_final = medoid_indices;
  labels = assignments;
}

/**
 * \brief Calculates confidence intervals in build step
 *
 * Calculates the confidence intervals about the reward for each arm
 *
 * @param data Transposed input data to find the medoids of
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param use_aboslute Determines whether the absolute cost is added to the total
 */
arma::rowvec BanditPAM::build_sigma(
  const arma::mat& data,
  const arma::rowvec& best_distances,
  bool use_absolute) {
    size_t N = data.n_cols;
    arma::uvec tmp_refs;
    // TODO(@motiwari): Make this wraparound properly as
    // last batch_size elements are dropped
    if (use_perm) {
      if ((permutation_idx + batchSize - 1) >= N) {
        permutation_idx = 0;
      }
      // inclusive of both indices
      tmp_refs = permutation.subvec(
        permutation_idx,
        permutation_idx + batchSize - 1);
      permutation_idx += batchSize;
    } else {
      tmp_refs = arma::randperm(N, batchSize);
    }

    arma::vec sample(batchSize);
    arma::rowvec updated_sigma(N);
    // for each possible swap
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        // gather a sample of points
        for (size_t j = 0; j < batchSize; j++) {
            double cost = KMedoids::cachedLoss(data, i, tmp_refs(j));
            if (use_absolute) {
                sample(j) = cost;
            } else {
                sample(j) = cost < best_distances(tmp_refs(j))
                              ? cost
                              : best_distances(tmp_refs(j));
                sample(j) -= best_distances(tmp_refs(j));
            }
        }
        updated_sigma(i) = arma::stddev(sample);
    }

    return updated_sigma;
}

/**
 * \brief Build step for BanditPAM
 *
 * Runs build step for the BanditPAM algorithm. Draws batch sizes with replacement
 * from reference set, and uses the estimated reward of the potential medoid
 * solutions on the reference set to update the reward confidence intervals and
 * accordingly narrow the solution set.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Uninitialized array of medoids that is modified in place
 * as medoids are identified
 * @param medoids Matrix of possible medoids that is updated as the bandit
 * learns which datapoints will be unlikely to be good candidates
 */
void BanditPAM::build(
  const arma::mat& data,
  arma::urowvec* medoid_indices,
  arma::mat* medoids) {
    size_t N = data.n_cols;
    arma::rowvec N_mat(N);
    N_mat.fill(N);
    size_t p = (buildConfidence * N);
    bool use_absolute = true;
    arma::rowvec estimates(N, arma::fill::zeros);
    arma::rowvec best_distances(N);
    best_distances.fill(std::numeric_limits<double>::infinity());
    arma::rowvec sigma(N);
    arma::urowvec candidates(N, arma::fill::ones);
    arma::rowvec lcbs(N);
    arma::rowvec ucbs(N);
    arma::rowvec T_samples(N, arma::fill::zeros);
    arma::rowvec exact_mask(N, arma::fill::zeros);

    for (size_t k = 0; k < n_medoids; k++) {
        // instantiate medoids one-by-online
        permutation_idx = 0;
        size_t step_count = 0;
        candidates.fill(1);
        T_samples.fill(0);
        exact_mask.fill(0);
        estimates.fill(0);
        // compute std dev amongst batch of reference points
        sigma = build_sigma(
                data, best_distances, use_absolute);

        while (arma::sum(candidates) > precision) {
            arma::umat compute_exactly =
              ((T_samples + batchSize) >= N_mat) != exact_mask;
            if (arma::accu(compute_exactly) > 0) {
                arma::uvec targets = find(compute_exactly);
                arma::rowvec result = build_target(
                  data,
                  &targets,
                  &best_distances,
                  use_absolute,
                  N);
                estimates.cols(targets) = result;
                ucbs.cols(targets) = result;
                lcbs.cols(targets) = result;
                exact_mask.cols(targets).fill(1);
                T_samples.cols(targets) += N;
                candidates.cols(targets).fill(0);
            }
            if (arma::sum(candidates) < precision) {
                break;
            }
            arma::uvec targets = arma::find(candidates);
            arma::rowvec result = build_target(
              data,
              &targets,
              &best_distances,
              use_absolute,
              0);
            // update the running average
            estimates.cols(targets) =
              ((T_samples.cols(targets) % estimates.cols(targets)) +
              (result * batchSize)) /
              (batchSize + T_samples.cols(targets));
            T_samples.cols(targets) += batchSize;
            arma::rowvec adjust(targets.n_rows);
            adjust.fill(p);
            adjust = arma::log(adjust);
            arma::rowvec cb_delta =
              sigma.cols(targets) %
              arma::sqrt(adjust / T_samples.cols(targets));
            ucbs.cols(targets) = estimates.cols(targets) + cb_delta;
            lcbs.cols(targets) = estimates.cols(targets) - cb_delta;
            candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            step_count++;
        }

        medoid_indices->at(k) = lcbs.index_min();
        medoids->unsafe_col(k) = data.unsafe_col((*medoid_indices)(k));

        // don't need to do this on final iteration
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            double cost = KMedoids::cachedLoss(data, i, (*medoid_indices)(k));
            if (cost < best_distances(i)) {
                best_distances(i) = cost;
            }
        }

        // use difference of loss for sigma and sampling, not absolute
        use_absolute = false;
    }
}

/**
 * \brief Estimates the mean reward for each arm in build step
 *
 * Estimates the mean reward (or loss) for each arm in the identified targets
 * in the build step and returns a list of the estimated reward.
 *
 * @param data Transposed input data to find the medoids of
 * @param target Set of target datapoints to be estimated
 *  intervals
 * @param best_distances Array of best distances from each point to previous
 *  set of medoids
 * @param use_absolute Determines whether the absolute cost is added to the total
 */
arma::rowvec BanditPAM::build_target(
  const arma::mat& data,
  arma::uvec* target,
  arma::rowvec* best_distances,
  bool use_absolute,
  size_t exact = 0) {
    size_t N = data.n_cols;
    size_t tmp_batch_size = batchSize;
    if (exact > 0) {
      tmp_batch_size = N;
    }
    arma::rowvec estimates(target->n_rows, arma::fill::zeros);
    arma::uvec tmp_refs;
    // TODO(@motiwari): Make this wraparound properly
    // as last batch_size elements are dropped
    if (use_perm) {
      if ((permutation_idx + tmp_batch_size - 1) >= N) {
        permutation_idx = 0;
      }
      // inclusive of both indices
      tmp_refs = permutation.subvec(
        permutation_idx,
        permutation_idx + tmp_batch_size - 1);
      permutation_idx += tmp_batch_size;
    } else {
      tmp_refs = arma::randperm(N, tmp_batch_size);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < target->n_rows; i++) {
        double total = 0;
        for (size_t j = 0; j < tmp_refs.n_rows; j++) {
            double cost =
              KMedoids::cachedLoss(data, (*target)(i), tmp_refs(j));
            if (use_absolute) {
                total += cost;
            } else {
                total += cost < (*best_distances)(tmp_refs(j))
                          ? cost
                          : (*best_distances)(tmp_refs(j));
                total -= (*best_distances)(tmp_refs(j));
            }
        }
        estimates(i) = total / tmp_batch_size;
    }
    return estimates;
}

/**
 * \brief Calculates confidence intervals in swap step
 *
 * Calculates the confidence intervals about the reward for each arm
 *
 * @param data Transposed input data to find the medoids of
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param second_best_distances Array of second smallest distances from each
 * point to previous set of medoids
 * @param assignments Assignments of datapoints to their closest medoid
 */
arma::mat BanditPAM::swap_sigma(
  const arma::mat& data,
  arma::rowvec* best_distances,
  arma::rowvec* second_best_distances,
  arma::urowvec* assignments) {
    size_t N = data.n_cols;
    size_t K = n_medoids;
    arma::mat updated_sigma(K, N, arma::fill::zeros);
    arma::uvec tmp_refs;
    // TODO(@motiwari): Make this wraparound properly
    // as last batch_size elements are dropped
    if (use_perm) {
      if ((permutation_idx + batchSize - 1) >= N) {
        permutation_idx = 0;
      }
      // inclusive of both indices
      tmp_refs = permutation.subvec(
        permutation_idx,
        permutation_idx + batchSize - 1);
      permutation_idx += batchSize;
    } else {
      tmp_refs = arma::randperm(N, batchSize);
    }

    arma::vec sample(batchSize);
    // for each considered swap
    #pragma omp parallel for
    for (size_t i = 0; i < K * N; i++) {
        // extract data point of swap
        size_t n = i / K;
        size_t k = i % K;

        // calculate change in loss for some subset of the data
        for (size_t j = 0; j < batchSize; j++) {
            double cost = KMedoids::cachedLoss(data, n, tmp_refs(j));

            if (k == (*assignments)(tmp_refs(j))) {
                if (cost < (*second_best_distances)(tmp_refs(j))) {
                    sample(j) = cost;
                } else {
                    sample(j) = (*second_best_distances)(tmp_refs(j));
                }
            } else {
                if (cost < (*best_distances)(tmp_refs(j))) {
                    sample(j) = cost;
                } else {
                    sample(j) = (*best_distances)(tmp_refs(j));
                }
            }
            sample(j) -= (*best_distances)(tmp_refs(j));
        }
        updated_sigma(k, n) = arma::stddev(sample);
    }
  return updated_sigma;
}

/**
 * \brief Swap step for BanditPAM
 *
 * Runs Swap step for the BanditPAM algorithm. Draws batch sizes with replacement
 * from reference set, and uses the estimated reward of the potential medoid
 * solutions on the reference set to update the reward confidence intervals and
 * accordingly narrow the solution set.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices created from the build step
 * that is modified in place as better medoids are identified
 * @param medoids Matrix of possible medoids that is updated as the bandit
 * learns which datapoints will be unlikely to be good candidates
 * @param assignments Uninitialized array of indices corresponding to each
 * datapoint assigned the index of the medoid it is closest to
 */
void BanditPAM::swap(
  const arma::mat& data,
  arma::urowvec* medoid_indices,
  arma::mat* medoids,
  arma::urowvec* assignments) {
    size_t N = data.n_cols;
    size_t p = (N * n_medoids * swapConfidence);

    arma::mat sigma(n_medoids, N, arma::fill::zeros);

    arma::rowvec best_distances(N);
    arma::rowvec second_distances(N);
    size_t iter = 0;
    bool swap_performed = true;
    arma::umat candidates(n_medoids, N, arma::fill::ones);
    arma::umat exact_mask(n_medoids, N, arma::fill::zeros);
    arma::mat estimates(n_medoids, N, arma::fill::zeros);
    arma::mat lcbs(n_medoids, N);
    arma::mat ucbs(n_medoids, N);
    arma::umat T_samples(n_medoids, N, arma::fill::zeros);

    // continue making swaps while loss is decreasing
    while (swap_performed && iter < max_iter) {
        iter++;
        permutation_idx = 0;

        // calculate quantities needed for swap, best_distances and sigma
        calc_best_distances_swap(
          data,
          medoid_indices,
          &best_distances,
          &second_distances,
          assignments);

        sigma = swap_sigma(data,
                          &best_distances,
                          &second_distances,
                          assignments);

        candidates.fill(1);
        exact_mask.fill(0);
        estimates.fill(0);
        T_samples.fill(0);

        // while there is at least one candidate (double comparison issues)
        while (arma::accu(candidates) > 0.5) {
            calc_best_distances_swap(
              data,
              medoid_indices,
              &best_distances,
              &second_distances,
              assignments);

            // compute exactly if it's been samples more than N times and
            // hasn't been computed exactly already
            arma::umat compute_exactly =
              ((T_samples + batchSize) >= N) != (exact_mask);
            arma::uvec targets = arma::find(compute_exactly);

            if (targets.size() > 0) {
                arma::vec result = swap_target(data,
                                              medoid_indices,
                                              &targets,
                                              &best_distances,
                                              &second_distances,
                                              assignments,
                                              N);
                estimates.elem(targets) = result;
                ucbs.elem(targets) = result;
                lcbs.elem(targets) = result;
                exact_mask.elem(targets).fill(1);
                T_samples.elem(targets) += N;

                candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            }
            if (arma::accu(candidates) < precision) {
                break;
            }
            targets = arma::find(candidates);
            arma::vec result = swap_target(data,
                                          medoid_indices,
                                          &targets,
                                          &best_distances,
                                          &second_distances,
                                          assignments,
                                          0);
            estimates.elem(targets) =
              ((T_samples.elem(targets) % estimates.elem(targets)) +
              (result * batchSize)) /
              (batchSize + T_samples.elem(targets));
            T_samples.elem(targets) += batchSize;
            arma::vec adjust(targets.n_rows);
            adjust.fill(p);
            adjust = arma::log(adjust);
            arma::vec cb_delta = sigma.elem(targets) %
                                arma::sqrt(adjust / T_samples.elem(targets));

            ucbs.elem(targets) = estimates.elem(targets) + cb_delta;
            lcbs.elem(targets) = estimates.elem(targets) - cb_delta;
            candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            targets = arma::find(candidates);
        }
        // now switch medoids
        arma::uword new_medoid = lcbs.index_min();
        // extract medoid of swap
        size_t k = new_medoid % medoids->n_cols;

        // extract data point of swap
        size_t n = new_medoid / medoids->n_cols;
        swap_performed = (*medoid_indices)(k) != n;
        steps++;

        (*medoid_indices)(k) = n;
        medoids->col(k) = data.col((*medoid_indices)(k));
        calc_best_distances_swap(
          data,
          medoid_indices,
          &best_distances,
          &second_distances,
          assignments);
    }
}

/**
 * \brief Estimates the mean reward for each arm in swap step
 *
 * Estimates the mean reward (or loss) for each arm in the identified targets
 * in the swap step and returns a list of the estimated reward.
 *
 * @param data Transposed input data to find the medoids of
 * @param targets Set of target datapoints to be estimated
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param second_best_distances Array of second smallest distances from each
 * point to previous set of medoids
 * @param assignments Assignments of datapoints to their closest medoid
 */
arma::vec BanditPAM::swap_target(
  const arma::mat& data,
  arma::urowvec* medoid_indices,
  arma::uvec* targets,
  arma::rowvec* best_distances,
  arma::rowvec* second_best_distances,
  arma::urowvec* assignments,
  size_t exact = 0) {
    size_t N = data.n_cols;
    arma::vec estimates(targets->n_rows, arma::fill::zeros);

    size_t tmp_batch_size = batchSize;
    if (exact > 0) {
      tmp_batch_size = N;
    }

    arma::uvec tmp_refs;
    // TODO(@motiwari): Make this wraparound properly
    // as last batch_size elements are dropped
    if (use_perm) {
      if ((permutation_idx + tmp_batch_size - 1) >= N) {
        permutation_idx = 0;
      }
      // inclusive of both indices
      tmp_refs = permutation.subvec(
        permutation_idx,
        permutation_idx + tmp_batch_size - 1);
      permutation_idx += tmp_batch_size;
    } else {
      tmp_refs = arma::randperm(N, tmp_batch_size);
    }

    // for each considered swap
    #pragma omp parallel for
    for (size_t i = 0; i < targets->n_rows; i++) {
        double total = 0;
        // extract data point of swap
        size_t n = (*targets)(i) / medoid_indices->n_cols;
        size_t k = (*targets)(i) % medoid_indices->n_cols;
        // calculate total loss for some subset of the data
        for (size_t j = 0; j < tmp_batch_size; j++) {
            double cost = KMedoids::cachedLoss(data, n, tmp_refs(j));
            if (k == (*assignments)(tmp_refs(j))) {
                if (cost < (*second_best_distances)(tmp_refs(j))) {
                    total += cost;
                } else {
                    total += (*second_best_distances)(tmp_refs(j));
                }
            } else {
                if (cost < (*best_distances)(tmp_refs(j))) {
                    total += cost;
                } else {
                    total += (*best_distances)(tmp_refs(j));
                }
            }
            total -= (*best_distances)(tmp_refs(j));
        }
        estimates(i) = total / tmp_refs.n_rows;
    }
    return estimates;
}
}  // namespace km
