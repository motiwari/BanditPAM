#include <kmedoids_pywrapper.hpp>
#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace km {
void KMedoidsWrapper::fit_sparse(py::object sparse_matrix, const std::string& loss_fn) {
    try {
        // Check if it's a scipy sparse matrix
        py::module sp_sparse = py::module::import("scipy.sparse");

        bool is_sparse = py::cast<bool>(sp_sparse.attr("issparse")(sparse_matrix));
        if (!is_sparse) {
            throw std::runtime_error("Input matrix is not a scipy sparse matrix");
        }

        // Convert to CSR format for efficient row access
        py::object csr_matrix = sparse_matrix.attr("tocsr")();

        // Get matrix dimensions
        py::tuple shape = csr_matrix.attr("shape").cast<py::tuple>();
        size_t n_rows = shape[0].cast<size_t>();
        size_t n_cols = shape[1].cast<size_t>();

        // Get the sparse matrix data
        py::array_t<float> data = csr_matrix.attr("data").cast<py::array_t<float>>();
        py::array_t<int> indices = csr_matrix.attr("indices").cast<py::array_t<int>>();
        py::array_t<int> indptr = csr_matrix.attr("indptr").cast<py::array_t<int>>();

        // Convert sparse matrix to dense armadillo matrix
        // Note: This is a compromise - truly sparse operations would require
        // significant changes to the core algorithm
        arma::fmat dense_matrix(n_rows, n_cols, arma::fill::zeros);

        auto data_ptr = static_cast<const float*>(data.data());
        auto indices_ptr = static_cast<const int*>(indices.data());
        auto indptr_ptr = static_cast<const int*>(indptr.data());

        for (size_t i = 0; i < n_rows; ++i) {
            int start = indptr_ptr[i];
            int end = indptr_ptr[i + 1];

            for (int j = start; j < end; ++j) {
                int col = indices_ptr[j];
                float val = data_ptr[j];
                dense_matrix(i, col) = val;
            }
        }

        // Store the original sparse format info for predict method
        this->is_sparse_input = true;
        this->sparse_n_rows = n_rows;
        this->sparse_n_cols = n_cols;

        // Call the regular fit method with dense matrix
        this->fit(dense_matrix, loss_fn, std::nullopt);

    } catch (const py::error_already_set& e) {
        throw std::runtime_error("Failed to process sparse matrix: " + std::string(e.what()));
    }
}

void KMedoidsWrapper::predict_sparse(py::object sparse_matrix_new) {
    if (!this->is_sparse_input) {
        throw std::runtime_error("Model was not fitted with sparse data. Use regular predict() method.");
    }

    try {
        py::module sp_sparse = py::module::import("scipy.sparse");

        bool is_sparse = py::cast<bool>(sp_sparse.attr("issparse")(sparse_matrix_new));
        if (!is_sparse) {
            throw std::runtime_error("Input matrix is not a scipy sparse matrix");
        }

        // Convert to CSR and then to dense for prediction
        py::object csr_matrix = sparse_matrix_new.attr("tocsr")();

        py::tuple shape = csr_matrix.attr("shape").cast<py::tuple>();
        size_t n_rows = shape[0].cast<size_t>();
        size_t n_cols = shape[1].cast<size_t>();

        if (n_cols != this->sparse_n_cols) {
            throw std::runtime_error("Number of features in new sparse data must match training data.");
        }

        py::array_t<float> data = csr_matrix.attr("data").cast<py::array_t<float>>();
        py::array_t<int> indices = csr_matrix.attr("indices").cast<py::array_t<int>>();
        py::array_t<int> indptr = csr_matrix.attr("indptr").cast<py::array_t<int>>();

        arma::fmat dense_matrix(n_rows, n_cols, arma::fill::zeros);

        auto data_ptr = static_cast<const float*>(data.data());
        auto indices_ptr = static_cast<const int*>(indices.data());
        auto indptr_ptr = static_cast<const int*>(indptr.data());

        for (size_t i = 0; i < n_rows; ++i) {
            int start = indptr_ptr[i];
            int end = indptr_ptr[i + 1];

            for (int j = start; j < end; ++j) {
                int col = indices_ptr[j];
                float val = data_ptr[j];
                dense_matrix(i, col) = val;
            }
        }

        // Call regular predict method
        this->predict(dense_matrix);

    } catch (const py::error_already_set& e) {
        throw std::runtime_error("Failed to process sparse matrix for prediction: " + std::string(e.what()));
    }
}
}  // namespace km