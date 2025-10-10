#include <kmedoids_pywrapper.hpp>
#include <carma>
#include <armadillo>
#include <iostream>
#include <limits>

void KMedoidsWrapper::predict(const arma::fmat &X_new) {
    // Check if model has been fitted
    if (this->medoidIndicesFinal.empty()) {
        throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
    }
    
    size_t n_new = X_new.n_rows;
    size_t n_features = X_new.n_cols;
    size_t n_medoids = this->medoidIndicesFinal.n_elem;
    
    // Validate input dimensions
    if (n_features != this->data.n_cols) {
        throw std::runtime_error("Number of features in new data must match training data.");
    }
    
    // Resize labels_predict to hold new predictions
    this->labels_predict.resize(n_new);
    
    // For each new data point, find the closest medoid
    for (size_t i = 0; i < n_new; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_medoid = 0;
        
        for (size_t j = 0; j < n_medoids; ++j) {
            size_t medoid_idx = this->medoidIndicesFinal(j);
            
            // Calculate distance between new point and medoid
            double distance = 0.0;
            std::string loss_fn = this->getLossFn();
            
            if (loss_fn == "L1" || loss_fn == "manhattan") {
                for (size_t k = 0; k < n_features; ++k) {
                    distance += std::abs(X_new(i, k) - this->data(medoid_idx, k));
                }
            } else if (loss_fn == "L2" || loss_fn == "euclidean") {
                for (size_t k = 0; k < n_features; ++k) {
                    double diff = X_new(i, k) - this->data(medoid_idx, k);
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
            } else if (loss_fn == "cosine") {
                double dot_product = 0.0;
                double norm_new = 0.0;
                double norm_medoid = 0.0;
                for (size_t k = 0; k < n_features; ++k) {
                    dot_product += X_new(i, k) * this->data(medoid_idx, k);
                    norm_new += X_new(i, k) * X_new(i, k);
                    norm_medoid += this->data(medoid_idx, k) * this->data(medoid_idx, k);
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