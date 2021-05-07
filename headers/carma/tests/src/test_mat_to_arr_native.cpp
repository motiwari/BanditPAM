#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <catch2/catch.hpp>

#include <carma>
namespace py = pybind11;

typedef arma::Mat<double> dMat;
typedef arma::Row<double> dRow;
typedef arma::Col<double> dCol;
typedef arma::Cube<double> dCube;
typedef py::array_t<double, py::array::f_style | py::array::forcecast> fArr;

TEST_CASE("Test mat_to_arr", "[mat_to_arr]") {
    SECTION("const l-value reference") {
        const dMat M = arma::randu<dMat>(100, 2);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::mat_to_arr<double>(M);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference with copy") {
        dMat M = arma::randu<dMat>(100, 2);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::mat_to_arr<double>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference without copy") {
        dMat M = arma::randu<dMat>(100, 2);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::mat_to_arr<double>(M, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::mat_to_arr<double>(arma::randu<dMat>(100, 2));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dMat M = arma::randu<dMat>(100, 2);
        dMat* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::mat_to_arr<double>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("pointer without copy") {
        dMat M = arma::randu<dMat>(100, 2);
        dMat* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::mat_to_arr<double>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }
} /* TEST_CASE MAT_TO_ARR */

TEST_CASE("Test row_to_arr", "[row_to_arr]") {
    SECTION("const l-value reference") {
        const dRow M = arma::randu<dRow>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::row_to_arr<double>(M);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference with copy") {
        dRow M = arma::randu<dRow>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::row_to_arr<double>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference without copy") {
        dRow M = arma::randu<dRow>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::row_to_arr<double>(M, false);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::row_to_arr<double>(arma::randu<dRow>(100));

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dRow M = arma::randu<dRow>(100);
        dRow* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::row_to_arr<double>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("pointer without copy") {
        dRow M = arma::randu<dRow>(100);
        dRow* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::row_to_arr<double>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }
} /* TEST_CASE ROW_TO_ARR */

TEST_CASE("Test col_to_arr", "[col_to_arr]") {
    SECTION("const l-value reference") {
        const dCol M = arma::randu<dCol>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::col_to_arr(M);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference with copy") {
        dCol M = arma::randu<dCol>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::col_to_arr(M, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference without copy") {
        dCol M = arma::randu<dCol>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::col_to_arr(M, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::col_to_arr<double>(arma::randu<dCol>(100));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dCol M = arma::randu<dCol>(100);
        dCol* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::col_to_arr<double>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("pointer without copy") {
        dCol M = arma::randu<dCol>(100);
        dCol* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::col_to_arr<double>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }
} /* TEST_CASE COL_TO_ARR */

TEST_CASE("Test cube_to_arr", "[cube_to_arr]") {
    SECTION("const l-value reference") {
        const dCube M = arma::randu<dCube>(100, 2, 4);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::cube_to_arr(M);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M_ptr);
    }

    SECTION("l-value reference with copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::cube_to_arr<double>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M_ptr);
    }

    SECTION("l-value reference without copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::cube_to_arr<double>(M, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::cube_to_arr<double>(arma::randu<dCube>(100, 2, 4));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);
        dCube* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::cube_to_arr<double>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M_ptr);
    }

    SECTION("pointer without copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);
        dCube* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::cube_to_arr<double>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M_ptr);
    }
} /* TEST_CASE CUBE_TO_ARR */

// ############################################################################
//                               TO_NUMPY
// ############################################################################

TEST_CASE("Test to_numpy Mat", "[to_numpy<Mat>]") {
    SECTION("const l-value reference") {
        const dMat M = arma::randu<dMat>(100, 2);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dMat>(M);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference with copy") {
        dMat M = arma::randu<dMat>(100, 2);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dMat>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference without copy") {
        dMat M = arma::randu<dMat>(100, 2);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dMat>(M, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::to_numpy<dMat>(arma::randu<dMat>(100, 2));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("r-value reference with long") {
        py::array_t<long> arr = carma::to_numpy<arma::Mat<long>>(arma::Mat<long>(100, 2, arma::fill::ones));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const long* ptr = static_cast<long*>(info.ptr);

        // compute sum of array
        long arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(arr_sum == 200);
    }

    SECTION("pointer with copy") {
        dMat M = arma::randu<dMat>(100, 2);
        dMat* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dMat>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("pointer without copy") {
        dMat M = arma::randu<dMat>(100, 2);
        dMat* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dMat>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 200);
        CHECK(arr_S0 == 100);
        CHECK(arr_S1 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }
} /* TEST_CASE MAT_TO_ARR */

TEST_CASE("Test to_numpy Row", "[to_numpy<Row>]") {
    SECTION("const l-value reference") {
        const dRow M = arma::randu<dRow>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dRow>(M);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference with copy") {
        dRow M = arma::randu<dRow>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dRow>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference without copy") {
        dRow M = arma::randu<dRow>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dRow>(M, false);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::to_numpy<dRow>(arma::randu<dRow>(100));

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dRow M = arma::randu<dRow>(100);
        dRow* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dRow>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("pointer without copy") {
        dRow M = arma::randu<dRow>(100);
        dRow* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dRow>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S1 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }
} /* TEST_CASE ROW_TO_ARR */

TEST_CASE("Test to_numpy Col", "[to_numpy<Col>]") {
    SECTION("const l-value reference") {
        const dCol M = arma::randu<dCol>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCol>(M);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference with copy") {
        dCol M = arma::randu<dCol>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCol>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("l-value reference without copy") {
        dCol M = arma::randu<dCol>(100);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCol>(M, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::to_numpy<dCol>(arma::randu<dCol>(100));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dCol M = arma::randu<dCol>(100);
        dCol* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCol>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr != M_ptr);
        CHECK(M_ptr == M.memptr());
        CHECK(std::abs(arma::accu(M) - mat_sum) < 1e-12);
    }

    SECTION("pointer without copy") {
        dCol M = arma::randu<dCol>(100);
        dCol* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCol>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t i = 0; i < arr_N; i++) {
            arr_sum += ptr[i];
        }

        // variable for test status
        CHECK(arr_N == 100);
        CHECK(arr_S0 == 100);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(ptr == M_ptr);
    }
} /* TEST_CASE COL_TO_ARR */

TEST_CASE("Test to_numpy Cube", "[to_numpy<Cube>]") {
    SECTION("const l-value reference") {
        const dCube M = arma::randu<dCube>(100, 2, 4);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCube>(M);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M_ptr);
    }

    SECTION("l-value reference with copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCube>(M, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M_ptr);
    }

    SECTION("r-value reference") {
        py::array_t<double> arr = carma::to_numpy<dCube>(arma::randu<dCube>(100, 2, 4));

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum) > 1e-12);
    }

    SECTION("pointer with copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);
        dCube* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCube>(src, true);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M_ptr);
    }

    SECTION("pointer without copy") {
        dCube M = arma::randu<dCube>(100, 2, 4);
        dCube* src = &M;

        double mat_sum = arma::accu(M);
        auto M_ptr = M.memptr();

        py::array_t<double> arr = carma::to_numpy<dCube>(src, false);

        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }
        // variable for test status
        CHECK(arr_N == 800);
        CHECK(arr_S0 == 4);
        CHECK(arr_S1 == 100);
        CHECK(arr_S2 == 2);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M_ptr);
    }
} /* TEST_CASE CUBE_TO_ARR */

// ############################################################################
//                               TO_NUMPY
// ############################################################################

TEST_CASE("Test update_array Mat", "[update_array<Mat>]") {
    py::module np_rand = py::module::import("numpy.random");
    py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

    dMat M = carma::arr_to_mat<double>(arr, false, false);

    M.insert_cols(0, 2, true);
    carma::update_array(M, arr);

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);

    double mat_sum = arma::accu(M);

    // compute sum of array
    double arr_sum = 0.0;
    auto p_arr = arr.unchecked<2>();
    for (size_t j = 0; j < arr_S1 ; j++) {
        for (size_t i = 0; i < arr_S0 ; i++) {
            arr_sum += p_arr(i, j);
        }
    }

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(arr_S1 == M.n_cols);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
} /* TEST_CASE UPDATE_ARR MAT */

TEST_CASE("Test update_array Row", "[update_array<Row>]") {
    py::module np_rand = py::module::import("numpy.random");
    py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

    dRow M = carma::arr_to_row<double>(arr, false, false);

    M.insert_cols(0, 2, true);
    carma::update_array(M, arr);

    double mat_sum = arma::accu(M);

    // get buffer for raw pointer
    py::buffer_info info = arr.request();
    const double* ptr = reinterpret_cast<double*>(info.ptr);

    // compute sum of array
    double arr_sum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
        arr_sum += ptr[i];

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(arr_S1 == M.n_cols);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
} /* TEST_CASE UPDATE_ARR ROW */

TEST_CASE("Test update_array Col", "[update_array<Col>]") {
    py::module np_rand = py::module::import("numpy.random");
    py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

    // get buffer for raw pointer
    dCol M = carma::arr_to_col<double>(arr, false, false);

    M.insert_rows(0, 2, true);
    carma::update_array(M, arr);

    double mat_sum = arma::accu(M);

    py::buffer_info info = arr.request();
    const double* ptr = reinterpret_cast<double*>(info.ptr);
    // compute sum of array
    double arr_sum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
        arr_sum += ptr[i];

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(arr_S1 == M.n_cols);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
} /* TEST_CASE UPDATE_ARR COL */

TEST_CASE("Test update_array Cube", "[update_array<Cube>]") {
    py::module np_rand = py::module::import("numpy.random");
    py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 1)));

    dCube M = carma::arr_to_cube<double>(arr, false, false);

    M.insert_cols(0, 2, true);
    carma::update_array(M, arr);

    double mat_sum = arma::accu(M);

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);
    size_t arr_S2 = arr.shape(2);

    // compute sum of array
    double arr_sum = 0.0;
    auto p_arr = arr.unchecked<3>();
    for (size_t si = 0; si < arr_S2; si++) {
        for (size_t ci = 0; ci < arr_S1; ci++) {
            for (size_t ri = 0; ri < arr_S0; ri++) {
                arr_sum += p_arr(ri, ci, si);
            }
        }
    }

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(arr_S1 == M.n_cols);
    CHECK(arr_S2 == M.n_slices);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
} /* TEST_CASE UPDATE_ARR CUBE */
