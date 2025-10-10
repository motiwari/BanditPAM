#include <kmedoids_pywrapper.hpp>
#include <carma>
#include <armadillo>
#include <iostream>

namespace km {
void KMedoidsWrapper::predict(const arma::fmat& X_new) {
    if (this->getMedoidsFinal().empty()) {
        throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
    }

    size_t n_new = X_new.n_rows;
    size_t n_features = X_new.n_cols;
    size_t n_medoids = this->getMedoidsFinal().size();

    // Validate input dimensions
    if (n_features != this->getData().n_cols) {
        throw std::runtime_error("Number of features in new data must match training data.");
    }

    // Resize labels_predict to hold new predictions
    this->labels_predict.resize(n_new);

    // For each new data point, find the closest medoid
    for (size_t i = 0; i < n_new; ++i) {
        float min_distance = std::numeric_limits<float>::max();
        size_t closest_medoid = 0;

        for (size_t j = 0; j < n_medoids; ++j) {
            size_t medoid_idx = this->getMedoidsFinal()[j];

            // Calculate distance between new point and medoid
            float distance = 0.0;
            if (this->getLossFn() == "L1" || this->getLossFn() == "manhattan") {
                for (size_t k = 0; k < n_features; ++k) {
                    distance += std::abs(X_new(i, k) - this->getData()(medoid_idx, k));
                }
            } else if (this->getLossFn() == "L2" || this->getLossFn() == "euclidean") {
                for (size_t k = 0; k < n_features; ++k) {
                    float diff = X_new(i, k) - this->getData()(medoid_idx, k);
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
            } else if (this->getLossFn() == "cosine") {
                float dot_product = 0.0;
                float norm_new = 0.0;
                float norm_medoid = 0.0;
                for (size_t k = 0; k < n_features; ++k) {
                    dot_product += X_new(i, k) * this->getData()(medoid_idx, k);
                    norm_new += X_new(i, k) * X_new(i, k);
                    norm_medoid += this->getData()(medoid_idx, k) * this->getData()(medoid_idx, k);
                }
                distance = 1.0 - (dot_product / (std::sqrt(norm_new) * std::sqrt(norm_medoid)));
            }

            if (distance < min_distance) {
                min_distance = distance;
                closest_medoid = j;
            }
        }

        this->labels_predict[i] = closest_medoid;
    }
}

const std::vector<size_t>& KMedoidsWrapper::get_predict_labels() const {
    if (this->labels_predict.empty()) {
        throw std::runtime_error("No predictions available. Call predict() first.");
    }
    return this->labels_predict;
}
}  // namespace km