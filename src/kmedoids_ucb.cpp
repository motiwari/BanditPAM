/**
 * @file kmedoids_ucb.cpp
 * @date 2020-06-09
 *
 * This file contains the implementation details for the confidence
 * bound improvement of the kmedoids PAM algorithim.
 */

#include "kmedoids_ucb.hpp"
// #include "pybind11.h"
//
// namespace py = pybind11

KMediods::KMediods(arma::mat data, size_t maxIterations, int verbosity, std::string loss): data(data), maxIterations(maxIterations), verbosity(verbosity) {
    // open filepointer if logging
    if (verbosity > 0) {
        std::string logName = "BanditPam_log"; // TODO: better logfile name
        logFile.open(logName);
    }
    logBuffer << "verbosity is " << verbosity << '\n';
    log(1);


    // set loss function
    if (loss == "manhattan") {
        lossFn = &KMediods::manhattan;
    } else if (loss == "cos") {
        lossFn = &KMediods::cos;
    } else if (loss == "L1") {
        lossFn = &KMediods::L1;
    } else if (loss == "L2"){
        lossFn = &KMediods::L2;
    } else {
        throw "unrecognized loss function";
    }

    // set loss function
    logBuffer << "loss function is " << loss << '\n';
    log(1);
}

KMediods::~KMediods() {
    if (verbosity > 0) {
        logFile.close();
    }
}

void
KMediods::cluster(const size_t clusters,
                  arma::urowvec& assignments,
                  arma::urowvec& medoid_indices)
{
    arma::mat medoids(data.n_rows, clusters);

    // build clusters
    logBuffer << "beginning build step" << '\n';
    log(1);
    KMediods::build(clusters, medoid_indices, medoids);
    logBuffer << "Medoid assignments:" << '\n';
    logBuffer << medoid_indices << '\n';
    log(1);

    // iterate swap steps
    logBuffer << "beginning swap step" << '\n';
    log(1);
    KMediods::swap(clusters, medoid_indices, medoids, assignments);
    logBuffer << "Medoid assignments:" << '\n';
    logBuffer << medoid_indices << '\n';
    log(2);
}

void
KMediods::build(
                const size_t clusters,
                arma::urowvec& medoid_indices,
                arma::mat& medoids)
{
    // Parameters
    size_t N = data.n_cols;
    arma::urowvec N_mat(N);
    N_mat.fill(N);
    int p = (k_buildConfidence * N); // reciprocal of
    bool use_absolute = true;
    arma::urowvec num_samples(N, arma::fill::zeros);
    arma::rowvec estimates(N, arma::fill::zeros);
    arma::rowvec best_distances(N);
    best_distances.fill(std::numeric_limits<double>::infinity());
    arma::rowvec sigma(N);
    arma::urowvec candidates(
      N,
      arma::fill::ones); // one hot encoding of candidates;
    arma::rowvec lcbs(N);
    arma::rowvec ucbs(N);
    arma::urowvec T_samples(N, arma::fill::zeros);
    arma::urowvec exact_mask(N, arma::fill::zeros);

    for (size_t k = 0; k < clusters; k++) {
        size_t step_count = 0;
        candidates.fill(1);
        T_samples.fill(0);
        exact_mask.fill(0);
        estimates.fill(0);
        KMediods::build_sigma(
           best_distances, sigma, k_batchSize, use_absolute);

        while (arma::sum(candidates) >
               k_doubleComparisonLimit) // double comparison
        {
            arma::umat compute_exactly =
              ((T_samples + k_batchSize) >= N_mat) != exact_mask;
            if (arma::accu(compute_exactly) > 0) {
                arma::uvec targets = find(compute_exactly);
                logBuffer << "Computing exactly for " << targets.n_rows
                          << " out of " << data.n_cols << '\n';
                log(2);
                arma::rowvec result =
                  build_target(targets, N, best_distances, use_absolute);
                estimates.cols(targets) = result;
                ucbs.cols(targets) = result;
                lcbs.cols(targets) = result;
                exact_mask.cols(targets).fill(1);
                T_samples.cols(targets) += N;
                candidates.cols(targets).fill(0);
            }
            if (arma::sum(candidates) < k_doubleComparisonLimit) {
                break;
            }
            arma::uvec targets = arma::find(candidates);

            arma::rowvec result = build_target(
              targets, k_batchSize, best_distances, use_absolute);
            estimates.cols(targets) =
              ((T_samples.cols(targets) % estimates.cols(targets)) +
               (result * k_batchSize)) /
              (k_batchSize + T_samples.cols(targets));
            T_samples.cols(targets) += k_batchSize;
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

        arma::uword new_medoid = lcbs.index_min();
        medoid_indices.at(k) = lcbs.index_min();
        medoids.unsafe_col(k) = data.unsafe_col(medoid_indices(k));

        // don't need to do this on final iteration
        for (int i = 0; i < N; i++) {
            double cost = (this->*lossFn)(i, medoid_indices(k));
            if (cost < best_distances(i)) {
                best_distances(i) = cost;
            }
        }
        use_absolute = false; // use difference of loss for sigma and sampling,
                              // not absolute
        logBuffer << "Loss: " << arma::mean(arma::mean(best_distances)) << '\n';
        log(2);
        logBuffer << "p: " << (float)1/(float)p << '\n';
        log(2);
    }
}

void
KMediods::build_sigma(
                      arma::rowvec& best_distances,
                      arma::rowvec& sigma,
                      arma::uword batch_size,
                      bool use_absolute)
{
    size_t N = data.n_cols;
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo
    arma::vec sample(batch_size);
// for each possible swap
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        // gather a sample of points
        for (size_t j = 0; j < batch_size; j++) {
            double cost = (this->*lossFn)(i,tmp_refs(j));
            if (use_absolute) {
                sample(j) = cost;
            } else {
                sample(j) = cost < best_distances(tmp_refs(j))
                              ? cost
                              : best_distances(tmp_refs(j));
                sample(j) -= best_distances(tmp_refs(j));
            }
        }
        sigma(i) = arma::stddev(sample);
    }
    arma::rowvec P = {0.25, 0.5, 0.75};
    arma::rowvec Q = arma::quantile(sigma, P);
    logBuffer << "Sigma: min: " << arma::min(sigma)
              << ", 25th: " << Q(0)
              << ", median: " << Q(1)
              << ", 75th: " << Q(2)
              << ", max: " << arma::max(sigma)
              << ", mean: " << arma::mean(sigma) << '\n';
    log(2);
}

// forcibly inline this in the future and directly write to estimates
arma::rowvec
KMediods::build_target(
                       arma::uvec& target,
                       size_t batch_size,
                       arma::rowvec& best_distances,
                       bool use_absolute)
{
    size_t N = data.n_cols;
    arma::rowvec estimates(target.n_rows, arma::fill::zeros);
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo
    double total = 0;
#pragma omp parallel for
    for (size_t i = 0; i < target.n_rows; i++) {
        double total = 0;
        for (size_t j = 0; j < tmp_refs.n_rows; j++) {
            double cost =
              (this->*lossFn)(tmp_refs(j),target(i));
            if (use_absolute) {
                total += cost;
            } else {
                total += cost < best_distances(tmp_refs(j))
                           ? cost
                           : best_distances(tmp_refs(j));
                total -= best_distances(tmp_refs(j));
            }
        }
        estimates(i) = total / batch_size;
    }
    return estimates;
}

void
KMediods::swap_sigma(
                     arma::mat& sigma,
                     size_t batch_size,
                     arma::rowvec& best_distances,
                     arma::rowvec& second_best_distances,
                     arma::urowvec& assignments)
{
    size_t N = data.n_cols;
    size_t K = sigma.n_rows;
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo

    arma::vec sample(batch_size);
// for each considered swap
#pragma omp parallel for
    for (size_t i = 0; i < K * N; i++) {
        // extract data point of swap
        size_t n = i / K;
        size_t k = i % K;

        // calculate change in loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++) {
            double cost = (this->*lossFn)(n,tmp_refs(j));

            if (k == assignments(tmp_refs(j))) {
                if (cost < second_best_distances(tmp_refs(j))) {
                    sample(j) = cost;
                } else {
                    sample(j) = second_best_distances(tmp_refs(j));
                }
            } else {
                if (cost < best_distances(tmp_refs(j))) {
                    sample(j) = cost;
                } else {
                    sample(j) = best_distances(tmp_refs(j));
                }
            }
            sample(j) -= best_distances(tmp_refs(j));
        }
        sigma(k, n) = arma::stddev(sample);
    }
    // arma::rowvec P = {0.25, 0.5, 0.75};
    // arma::rowvec Q = arma::quantile(sigma, P);
    // logBuffer << "Sigma: min: " << sigma.min()
    //           << ", 25th: " << Q(0)
    //           << ", median: " << Q(1)
    //           << ", 75th: " << Q(2)
    //           << ", max: " << sigma.max() << '\n';
    //           // << ", mean: " << arma::mean(arma::mean(sigma)) << '\n';
    // log(2);
}

arma::vec
KMediods::swap_target(
                      arma::urowvec& medoid_indices,
                      arma::uvec& targets,
                      size_t batch_size,
                      arma::rowvec& best_distances,
                      arma::rowvec& second_best_distances,
                      arma::urowvec& assignments)
{
    size_t N = data.n_cols;
    arma::vec estimates(targets.n_rows, arma::fill::zeros);
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo

// for each considered swap
#pragma omp parallel for
    for (size_t i = 0; i < targets.n_rows; i++) {
        double total = 0;
        // extract data point of swap
        size_t n = targets(i) / medoid_indices.n_cols;
        size_t k = targets(i) % medoid_indices.n_cols;
        // calculate total loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++) {
            double cost = (this->*lossFn)(n, tmp_refs(j));
            if (k == assignments(tmp_refs(j))) {
                if (cost < second_best_distances(tmp_refs(j))) {
                    total += cost;
                } else {
                    total += second_best_distances(tmp_refs(j));
                }
            } else {
                if (cost < best_distances(tmp_refs(j))) {
                    total += cost;
                } else {
                    total += best_distances(tmp_refs(j));
                }
            }
            total -= best_distances(tmp_refs(j));
        }
        estimates(i) = total / tmp_refs.n_rows;
    }
    return estimates;
}

void
KMediods::calc_best_distances_swap(
                         arma::urowvec& medoid_indices,
                         arma::rowvec& best_distances,
                         arma::rowvec& second_distances,
                         arma::urowvec& assignments)
{
#pragma omp parallel for
    for (size_t i = 0; i < data.n_cols; i++) {
        double best = std::numeric_limits<double>::infinity();
        double second = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < medoid_indices.n_cols; k++) {
            double cost = (this->*lossFn)(medoid_indices(k), i);
            if (cost < best) {
                assignments(i) = k;
                second = best;
                best = cost;
            } else if (cost < second) {
                second = cost;
            }
        }
        best_distances(i) = best;
        second_distances(i) = second;
    }
}

void
KMediods::swap(
               const size_t clusters,
               arma::urowvec& medoid_indices,
               arma::mat& medoids,
               arma::urowvec& assignments)
{
    size_t N = data.n_cols;
    int p = (N * clusters * k_swapConfidence); // reciprocal

    arma::mat sigma(clusters, N, arma::fill::zeros);

    arma::rowvec best_distances(N);
    arma::rowvec second_distances(N);
    size_t iter = 0;
    bool swap_performed = true;
    arma::umat candidates(clusters, N, arma::fill::ones);
    arma::umat exact_mask(clusters, N, arma::fill::zeros);
    arma::mat estimates(clusters, N, arma::fill::zeros);
    arma::mat lcbs(clusters, N);
    arma::mat ucbs(clusters, N);
    arma::umat T_samples(clusters, N, arma::fill::zeros);

    // continue making swaps while loss is decreasing
    while (swap_performed && iter < maxIterations) {
        iter++;

        // calculate quantities needed for swap, best_distances and sigma
        calc_best_distances_swap(
          medoid_indices, best_distances, second_distances, assignments);

        swap_sigma(sigma,
                   k_batchSize,
                   best_distances,
                   second_distances,
                   assignments);

        candidates.fill(1);
        exact_mask.fill(0);
        estimates.fill(0);
        T_samples.fill(0);

        size_t step_count = 0;

        // while there is at least one candidate (double comparison issues)
        while (arma::accu(candidates) > 0.5) {
            calc_best_distances_swap(
              medoid_indices, best_distances, second_distances, assignments);

            // compute exactly if it's been samples more than N times and hasn't
            // been computed exactly already
            arma::umat compute_exactly =
              ((T_samples + k_batchSize) >= N) != (exact_mask);
            arma::uvec targets = arma::find(compute_exactly);

            if (targets.size() > 0) {
                size_t n = targets(0) / medoids.n_cols;
                size_t k = targets(0) % medoids.n_cols;
                logBuffer << "COMPUTING EXACTLY " << targets.size()
                          << " out of " << candidates.size() << '\n';
                log(2);
                arma::vec result = swap_target(medoid_indices,
                                               targets,
                                               N,
                                               best_distances,
                                               second_distances,
                                               assignments);
                estimates.elem(targets) = result;
                ucbs.elem(targets) = result;
                lcbs.elem(targets) = result;
                exact_mask.elem(targets).fill(1);
                T_samples.elem(targets) += N;

                candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            }
            if (arma::accu(candidates) < k_doubleComparisonLimit) {
                break;
            }
            targets = arma::find(candidates);
            arma::vec result = swap_target(medoid_indices,
                                           targets,
                                           k_batchSize,
                                           best_distances,
                                           second_distances,
                                           assignments);
            estimates.elem(targets) =
              ((T_samples.elem(targets) % estimates.elem(targets)) +
               (result * k_batchSize)) /
              (k_batchSize + T_samples.elem(targets));
            T_samples.elem(targets) += k_batchSize;
            arma::vec adjust(targets.n_rows);
            adjust.fill(p);
            adjust = arma::log(adjust);
            arma::vec cb_delta = sigma.elem(targets) %
                                 arma::sqrt(adjust / T_samples.elem(targets));

            ucbs.elem(targets) = estimates.elem(targets) + cb_delta;
            lcbs.elem(targets) = estimates.elem(targets) - cb_delta;
            candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            targets = arma::find(candidates);
            step_count++;
        }
        // now switch medoids
        arma::uword new_medoid = lcbs.index_min();
        // extract medoid of swap
        size_t k = new_medoid % medoids.n_cols;

        // extract data point of swap
        size_t n = new_medoid / medoids.n_cols;
        swap_performed = medoid_indices(k) != n;
            logBuffer << (swap_performed ? ("swap performed")
                                         : ("no swap performed"))
                      << " " << medoid_indices(k) << "to" << n << '\n';
        log(2);
        medoid_indices(k) = n;
        medoids.col(k) = data.col(medoid_indices(k));
        calc_best_distances_swap(
          medoid_indices, best_distances, second_distances, assignments);
        arma::rowvec P = {0.25, 0.5, 0.75};
        // arma::rowvec Q = arma::quantile(sigma.elem(targets), P);
        logBuffer << "Sigma: min: " << sigma.min()
        // << ", 25th: " << Q(0)
        // << ", median: " << Q(1)
        // << ", 75th: " << Q(2)
        << ", max: " << sigma.max()
        << ", mean: " << arma::mean(arma::mean(sigma)) << '\n';
        log(2);
        logBuffer << "Loss: " << arma::mean(arma::mean(best_distances)) << '\n';
        log(2);
        logBuffer << "p: " << (float)1/(float)p << '\n';
        log(2);
    }
    // logBuffer << "final swap loss: " << arma::mean(arma::mean(best_distances))
    //           << '\n';
    // log(1);

    // done with swaps at this point
}

double
KMediods::calc_loss(
                    const size_t clusters,
                    arma::urowvec& medoid_indices)
{
    double total = 0;

    for (size_t i = 0; i < data.n_cols; i++) {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < clusters; k++) {
            double currCost = (this->*lossFn)(medoid_indices(k), i);
            if (currCost < cost) {
                cost = currCost;
            }
        }
        total += cost;
    }
    return total;
}

void KMediods::log(int priority) {
    // if it won't be logged
    if (priority > verbosity) {
        logFile << logBuffer.rdbuf();
    }
    logFile.clear();
}

double KMediods::Lp(int i, int j) const {
  return arma::norm(data.col(i) - data.col(j), 1);
}

double KMediods::L1(int i, int j) const {
    return arma::norm(data.col(i) - data.col(j), 1);
}

double KMediods::L2(int i, int j) const {
    return arma::norm(data.col(i) - data.col(j), 2);
}

double KMediods::cos(int i, int j) const {
    return arma::dot(data.col(i), data.col(j)) / (arma::norm(data.col(i)) * arma::norm(data.col(j)));
}

double KMediods::manhattan(int i, int j) const {
    return arma::accu(arma::abs(data.col(i) - data.col(j)));
}

// PYBIND11_MODULE(bpam, m) {
//     py::class_<KMediods>(m, "kmeds")
//         .def(py::init<const st::string &name>())
//         .def("cluster", &KMediods::cluster);
// }
