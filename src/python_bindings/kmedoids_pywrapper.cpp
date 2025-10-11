/*
 * This file is part of BanditPAM.
 * 
 * BanditPAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * BanditPAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "kmedoids_pywrapper.hpp"

namespace py = pybind11;

// Forward declarations for all binding functions
void medoids_python(py::class_<km::KMedoidsWrapper> *cls);
void build_medoids_python(py::class_<km::KMedoidsWrapper> *cls);
void loss_python(py::class_<km::KMedoidsWrapper> *cls);
void build_loss_python(py::class_<km::KMedoidsWrapper> *cls);
void distance_computations_python(py::class_<km::KMedoidsWrapper> *cls);
void misc_distance_computations_python(py::class_<km::KMedoidsWrapper> *cls);
void cache_python(py::class_<km::KMedoidsWrapper> *cls);
void time_per_swap_python(py::class_<km::KMedoidsWrapper> *cls);
void total_swap_time_python(py::class_<km::KMedoidsWrapper> *cls);

namespace km {

PYBIND11_MODULE(banditpam, m) {
    m.doc() = "BanditPAM: Almost Linear-Time k-Medoids Clustering";
    
    py::class_<KMedoidsWrapper> cls(m, "KMedoids");
    
    // Constructor
    cls.def(py::init<int, const std::string&>(),
            "Constructor for KMedoids",
            py::arg("n_medoids"),
            py::arg("algorithm") = "BanditPAM");
    
    // Fit method
    cls.def("fit",
            [](KMedoidsWrapper &self,
               const py::array_t<float> &X,
               const std::string &loss_fn) {
                auto X_mat = carma::arr_to_mat<float>(X);
                self.fit(X_mat, loss_fn);
            },
            "Fit K-Medoids model to data",
            py::arg("X"),
            py::arg("loss_fn"));
    
    // Properties
    cls.def_property_readonly("medoids",
                             &KMedoidsWrapper::getMedoidsFinal,
                             "Final medoid indices");
    
    cls.def_property_readonly("labels",
                             &KMedoidsWrapper::getLabels,
                             "Cluster labels for each data point");
    
    cls.def_property_readonly("average_loss",
                             &KMedoidsWrapper::getAverageLoss,
                             "Average loss of the clustering");
    
    cls.def_property_readonly("n_medoids",
                             &KMedoidsWrapper::getNMedoids,
                             "Number of medoids");

    // Binding for all other functions
    medoids_python(&cls);
    build_medoids_python(&cls);
    loss_python(&cls);
    build_loss_python(&cls);

    // Predict function binding
    cls.def("predict",
            [](KMedoidsWrapper &self,
               const py::array_t<float> &X_new) {
                auto X_new_mat = carma::arr_to_mat<float>(X_new);
                self.predict(X_new_mat);
            },
            "Predict cluster labels for new data points",
            py::arg("X_new"));

    cls.def_property_readonly("labels_predict",
                             &KMedoidsWrapper::get_predict_labels,
                             "Cluster labels for predicted data points");

    // Cache functions
    distance_computations_python(&cls);
    misc_distance_computations_python(&cls);
    cache_python(&cls);

    // Swap timing functions
    time_per_swap_python(&cls);
    total_swap_time_python(&cls);

    // Sparse matrix support
    cls.def("fit_sparse",
            &KMedoidsWrapper::fit_sparse,
            "Fit K-Medoids model to sparse data");
    cls.def("predict_sparse",
            &KMedoidsWrapper::predict_sparse,
            "Predict cluster labels for new sparse data points");
}

} // namespace km