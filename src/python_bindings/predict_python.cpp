#include <kmedoids_pywrapper.hpp>
#include <carma>
#include <armadillo>
#include <iostream>

void KMedoidsPyWrapper::predict(const arma::Mat<double> &X_new) {
    if (this->medoids_final.empty()) {
        throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
    }
    
    size_t n_new = X_new.n_rows;
    size_t n_features = X_new.n_cols;
    size_t n_medoids = this->medoids_final.size();
    
    // Validate input dimensions
    if (n_features != this->input_data.n_cols) {
        throw std::runtime_error("Number of features in new data must match training data.");
    }
    
    // Resize labels_predict to hold new predictions
    this->labels_predict.resize(n_new);
    
    // For each new data point, find the closest medoid
    for (size_t i = 0; i < n_new; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_medoid = 0;
        
        for (size_t j = 0; j < n_medoids; ++j) {
            size_t medoid_idx = this->medoids_final[j];
            
            // Calculate distance between new point and medoid
            double distance = 0.0;
            if (this->loss_fn == "L1" || this->loss_fn == "manhattan") {
                for (size_t k = 0; k < n_features; ++k) {
                    distance += std::abs(X_new(i, k) - this->input_data(medoid_idx, k));
                }
            } else if (this->loss_fn == "L2" || this->loss_fn == "euclidean") {
                for (size_t k = 0; k < n_features; ++k) {
                    double diff = X_new(i, k) - this->input_data(medoid_idx, k);
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
            } else if (this->loss_fn == "cosine") {
                double dot_product = 0.0;
                double norm_new = 0.0;
                double norm_medoid = 0.0;
                for (size_t k = 0; k < n_features; ++k) {
                    dot_product += X_new(i, k) * this->input_data(medoid_idx, k);
                    norm_new += X_new(i, k) * X_new(i, k);
                    norm_medoid += this->input_data(medoid_idx, k) * this->input_data(medoid_idx, k);
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

const std::vector<size_t>& KMedoidsPyWrapper::get_predict_labels() const {
    if (this->labels_predict.empty()) {
        throw std::runtime_error("No predictions available. Call predict() first.");
    }
    return this->labels_predict;
}