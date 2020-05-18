#include "kmedoids_ucb.hpp"

using namespace arma;

KMediods::KMediods(size_t maxIterations)
{
    this->maxIterations = maxIterations;
}

void KMediods::cluster(const arma::mat &data,
                       const size_t clusters,
                       arma::Row<size_t> &assignments)
{
    arma::mat medoids(data.n_rows, clusters);
    arma::Row<size_t> medoid_indicies(clusters);

    // build clusters
    KMediods::build(data, clusters, medoid_indicies, medoids);
    std::cout << medoid_indicies << std::endl;
    std::cout << "data size, rows:" << data.n_rows << " cols:" << data.n_cols << std::endl;
    std::cout << "########################### SWAP ##############################" << std::endl;

    KMediods::swap(data, clusters, medoid_indicies, medoids);
    std::cout << medoid_indicies << std::endl;


    /*size_t i = 0;
    bool medoid_change = true;
    while (i < maxIterations && medoid_change)
    {
        auto previous(centroid_indicies);
        KMediods::swap(data, clusters, assignments, centroid_indicies);
        std::cout << centroid_indicies << std::endl;
        medoid_change = arma::any(centroid_indicies != previous);
        //medoid_change = true;
        std::cout << "mediod change is " << medoid_change << std::endl;
        i++;
    }*/
    // swap thingy
}

void KMediods::build(const arma::mat &data,
                     const size_t clusters,
                     arma::Row<size_t> &medoid_indicies,
                     arma::mat &medoids)
{
    // Parameters
    size_t N = data.n_cols;
    arma::Row<size_t> N_mat(N);
    N_mat.fill(N);
    double p = 1 / (N * 10);
    arma::Row<size_t> num_samples(N, arma::fill::zeros);
    arma::Row<double> estimates(N, arma::fill::zeros);

    arma::Row<double> best_distances(N);
    best_distances.fill(std::numeric_limits<double>::infinity());

    for (size_t k = 0; k < clusters; k++)
    {
        size_t step_count = 0;
        arma::urowvec candidates(N, arma::fill::ones); // one hot encoding of candidates;
        arma::Row<double> lcbs(N);
        arma::Row<double> ucbs(N);
        lcbs.fill(1000);
        ucbs.fill(1000);
        arma::Row<size_t> T_samples(N, arma::fill::zeros);
        arma::Row<size_t> exact_mask(N, arma::fill::zeros);

        size_t original_batch_size = 20;
        size_t base = 1;

        while (arma::sum(candidates) > 0.1)
        {
            std::cout << "Step count" << step_count << std::endl;
            size_t this_batch_size = original_batch_size; //need to add scaling batch size

            arma::umat compute_exactly = (T_samples + this_batch_size) >= N_mat;

            compute_exactly = compute_exactly != exact_mask; //check this
            if (arma::accu(compute_exactly) > 0)
            {
                uvec targets = find(compute_exactly);
                std::cout << "Computing exactly for " << targets.n_rows << " on step count " << step_count << std::endl;


                arma::Row<double> result = build_target(data, targets, N, best_distances);
                std::cout << "setting estimates" << std::endl;

                estimates.cols(targets) = result;
                std::cout << "setting ucbs" << std::endl;

                ucbs.cols(targets) = result;
                std::cout << "setting lcbs" << std::endl;

                lcbs.cols(targets) = result;

                exact_mask.cols(targets).fill(1);
                T_samples.cols(targets) += N;
                candidates.cols(targets).fill(0);
            }
            std::cout << "was able to compute exactly" << std::endl;

            if (sum(candidates) < 0.5)
            {
                continue;
            }
            uvec targets = find(candidates);
            arma::Row<double> result = build_target(data, targets, this_batch_size, best_distances);
            estimates.cols(targets) = ((T_samples.cols(targets) % estimates.cols(targets)) + (result * this_batch_size)) / (this_batch_size + T_samples.cols(targets));
            T_samples.cols(targets) += this_batch_size;
            arma::Row<double> adjust(targets.n_rows);
            adjust.fill(std::log(1 / p));
            arma::Row<double> cb_delta = sigma * arma::sqrt(adjust / T_samples.cols(targets));

            ucbs.cols(targets) = estimates.cols(targets) + cb_delta;
            lcbs.cols(targets) = estimates.cols(targets) - cb_delta;

            candidates = (lcbs < ucbs.min()) != exact_mask;
            step_count++;
        }

        uword new_medoid = lcbs.index_min();
        medoid_indicies(k) = lcbs.index_min();
        medoids.col(k) = data.col(medoid_indicies(k));

        // don't need to do this on final iteration
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j <= k; j++)
            {
                double cost = norm(data.col(i) - data.col(medoid_indicies(j)), 2);
                if (cost < best_distances(i))
                {
                    best_distances(i) = cost;
                }
            }
        }
        std::cout << "found new medoid" << new_medoid << std::endl;
    }
}

// forcibly inline this in the future and directly write to estimates
arma::Row<double> KMediods::build_target(const arma::mat &data, arma::uvec &target, size_t batch_size, arma::Row<double> &best_distances)
{
    size_t N = data.n_cols;
    arma::Row<double> estimates(target.n_rows, arma::fill::zeros);
    //uvec tmp_refs = randi<uvec>(batch_size, distr_param(0, N - 1)); //with replacement
    uvec tmp_refs = randperm(N, batch_size); //without replacement, requires updated version of armadillo
    double total = 0;
    for (size_t i = 0; i < target.n_rows; i++)
    {
        estimates(i) = KMediods::cost_fn_build(data, target(i), tmp_refs, best_distances);
    }
    return estimates;
}

double KMediods::cost_fn_build(const arma::mat &data, arma::uword target, arma::uvec &tmp_refs, arma::Row<double> &best_distances)
{
    double total = 0;
    for (size_t i = 0; i < tmp_refs.n_rows; i++)
    {
        double cost = norm(data.col(tmp_refs(i)) - data.col(target), 2);
        total += best_distances(tmp_refs(i)) > cost ? cost : best_distances(tmp_refs(i));
    }
    return total;
}

arma::vec KMediods::swap_target(
    const arma::mat &data,
    const arma::mat &medoids,
    arma::uvec &targets,
    size_t batch_size,
    arma::Row<double> &best_distances,
    arma::Row<double> &second_best_distances,
    arma::urowvec &assignments)
{
    // targets is a vector containing integers. For some integer n
    // data point is n % 
    // targets, data, medoids, best_distances
    // pick 20 random points
    // for each target point
    //      total = 0
    //      for each ref point
    //          total += best_distance
    size_t N = data.n_cols;
    std::cout << "targets size " << targets.n_rows << std::endl;
    arma::vec estimates(targets.n_rows, arma::fill::zeros);
    uvec tmp_refs = arma::randperm(N, batch_size); //without replacement, requires updated version of armadillo

    // for each considered swap
    for (size_t i = 0; i < targets.n_rows; i++)
    {
        double total = 0;
        // extract medoid of swap
        //size_t k = targets(i) % medoids.n_cols;
            
        // extract data point of swap
        size_t k = targets(i) / N;
        size_t n = targets(i) - (k * N);
        //std::cout << "testing setting data point " << n << " to medoid " << k << std::endl;
        
        // calculate total loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++)
        {

            double cost = arma::norm(data.col(n) - data.col(tmp_refs(j)), 2);
            
            // the swap makes a better medoid
            if (cost < best_distances(tmp_refs(j)))
            {
                total += cost;
            }

            // the swap is not a better mediod, and replaced
            // the previous medoid
            else if (k == assignments(n))
            {
                total += second_best_distances(tmp_refs(j));
            }

            // the swap is not the current medoid and it not better
            // than the current medoid
            else
            {
                total += best_distances(tmp_refs(j));
            }
        }
        // total currently depends on the batch size, which seems distinctly wrong. maybe.
        estimates(i) = total;
        //std::cout << "estimate " << i << " :" << estimates(i) << std::endl;
    }
    return estimates;
}

void get_best_distances(
    const arma::mat &data, 
    const arma::mat &medoids,
    arma::Row<double> &best_distances,
    arma::Row<double> &second_distances,
    arma::urowvec &assignments)
{
    for (size_t i = 0; i < data.n_cols; i++)
    {
        double best = std::numeric_limits<double>::infinity();
        double second;
        for (size_t k = 0; k < medoids.n_cols; k++)
        {
            double cost = arma::norm(data.col(i) - medoids.col(k), 2);
            if (cost < best)
            {
                assignments(i) = k;
                second = best;
                best = cost;
            }
            else if (cost < second)
            {
                second = cost;
            }
        }
        best_distances(i) = best;
        second_distances(i) = second;
    }
}

void KMediods::swap(const arma::mat &data,
                    const size_t clusters,
                    arma::Row<size_t> &medoid_indicies,
                    arma::mat &medoids)
{
    size_t N = data.n_cols;
    double p = (N * clusters * 1000);
    arma::Row<double> best_distances(N);
    arma::Row<double> second_distances(N);
    arma::urowvec assignments(N);

    // does this need to be calculated in every iteration?
    get_best_distances(data, medoids, best_distances, second_distances, assignments);
    double loss = arma::mean(best_distances);
    size_t iter = 0;
    bool swap_performed = true;
    while (swap_performed && iter < maxIterations)
    {
        iter++;
        std::cout << iter << std::endl;
        // one hot encoding of candidates to try
        arma::umat candidates(clusters, N, arma::fill::ones);
        arma::mat estimates(clusters, N);
        arma::mat lcbs(clusters, N);
        arma::mat ucbs(clusters, N);
        lcbs.fill(0);
        ucbs.fill(1000);

        arma::umat T_samples(clusters, N, arma::fill::zeros);
        arma::umat exact_mask(clusters, N, arma::fill::zeros);

        size_t original_batch_size = 20;

        size_t step_count = 0;
        // while there is at least one candidate (double comparison issues)
        while (arma::accu(candidates) > 0.5)
        {
            std::cout << T_samples << std::endl;
            // batch size is currently constant
            size_t this_batch_size = original_batch_size;

            // compute exactly if it's been samples more than N times and hasn't been computed exactly already
            arma::umat compute_exactly = ((T_samples + this_batch_size) >= N) != exact_mask;
            arma::uvec targets = arma::find(compute_exactly);
            std::cout << "candidates size " << arma::accu(candidates) << std::endl;
            std::cout << "exact mask size " << arma::accu(exact_mask) << std::endl;
            std::cout << "samples size " << arma::accu(T_samples) << std::endl;
            if (targets.n_rows > 0)
            {
                std::cout << "COMPUTING EXACTLY " << targets.n_rows << std::endl;
                arma::vec result = swap_target(data, medoids, targets, N, best_distances, second_distances, assignments);
                estimates.elem(targets) = result;
                ucbs.elem(targets) = result;
                lcbs.elem(targets) = result;
                exact_mask.elem(targets).fill(1);
                T_samples.elem(targets) += N;

                candidates = (lcbs < ucbs.min()) && (exact_mask == 0);


            // this is my own modification?
            }
            if (arma::accu(candidates) < 0.5)
            {
                std::cout << "yeeting away" << std::endl;
                break;
            }
            targets = arma::find(candidates);
            arma::vec result = swap_target(data, medoids, targets, this_batch_size, best_distances, second_distances, assignments);
            estimates.elem(targets) = ((T_samples.elem(targets) % estimates.elem(targets)) + (result * this_batch_size)) / (this_batch_size + T_samples.elem(targets));
            T_samples.elem(targets) += this_batch_size;
            //std::cout << "estimates:" << arma::mean(arma::mean(estimates)) << std::endl;
            arma::vec adjust(targets.n_rows);
            //std::cout << "result of log " << ::log(p) << std::endl;
            adjust.fill(::log(p));
            arma::vec cb_delta = sigma * arma::sqrt(adjust / T_samples.elem(targets));
            //std::cout << "cb_delta:" << arma::mean(arma::mean(cb_delta)) << std::endl;

            ucbs.elem(targets) = estimates.elem(targets) + cb_delta;
            lcbs.elem(targets) = estimates.elem(targets) - cb_delta;
            //std::cout << "ucbs:" << arma::mean(arma::mean(ucbs)) << " lcbs:" << arma::mean(arma::mean(lcbs)) << std::endl;
            candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            targets = arma::find(candidates);
            step_count++;
        }
        // now switch medoids
        uword new_medoid = lcbs.index_min();
        // extract medoid of swap
        size_t k = new_medoid % medoids.n_cols;
            
        // extract data point of swap
        size_t n = new_medoid - (N * (new_medoid / N));
        swap_performed = medoid_indicies(k) != n;
        std::cout << (swap_performed ? ("swap performed") : ("no swap performed")) << std::endl;
        medoid_indicies(k) = n;
        medoids.col(k) = data.col(medoid_indicies(k));

    }
    // done with swaps at this point
}

double KMediods::calc_loss(const arma::mat &data,
                           const size_t clusters,
                           arma::Row<size_t> &centroid_indicies)
{
    double total = 0;

    for (size_t i = 0; i < data.n_cols; i++)
    {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t mediod = 0; mediod < clusters; mediod++)
        {
            if (arma::norm(data.col(centroid_indicies(mediod)) - data.col(i), 2) < cost)
            {
                cost = arma::norm(data.col(centroid_indicies(mediod)) - data.col(i), 2);
            }
        }
        total += cost;
    }
    return total;
}
