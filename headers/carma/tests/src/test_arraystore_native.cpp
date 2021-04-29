#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <catch2/catch.hpp>

#include <carma>
namespace py = pybind11;

typedef arma::Mat<long> lMat;
typedef arma::Mat<double> dMat;
typedef arma::Row<double> dRow;
typedef arma::Col<double> dCol;
typedef arma::Cube<double> dCube;

TEST_CASE("Test ArrayStore Mat", "[ArrayStore<Mat>]") {
    SECTION("const l-value constructor") {
        const dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.n_cols == out_S1);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value constructor -- copy") {
        dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.n_cols == out_S1);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value constructor -- steal") {
        dMat in = arma::randu<dMat>(100, 2);
        double in_sum = arma::accu(in);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in, false);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(100 == out.shape(0));
        CHECK(2 == out.shape(1));
        CHECK(std::abs(in_sum - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION(" r-value constructor") {
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(arma::randu<dMat>(100, 2));
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(100 == out.shape(0));
        CHECK(2 == out.shape(1));
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("get_view -- not writeable") {
        dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.n_cols == out_S1);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
        CHECK(carma::is_writeable(out) == false);
        CHECK_THROWS(out.mutable_unchecked());
    }

    SECTION("get_view -- not writeable") {
        dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in, true);
        py::array_t<double> out = store.get_view(true);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.n_cols == out_S1);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
        CHECK(carma::is_writeable(out) == true);
    }

    SECTION("const l-value set_data") {
        const dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in);

        const dMat sec = arma::randu<dMat>(100, 2);
        store.set_data(sec);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(sec.n_elem == out_N);
        CHECK(sec.n_rows == out_S0);
        CHECK(sec.n_cols == out_S1);
        CHECK(sec.memptr() != ptr);
        CHECK(std::abs(arma::accu(sec) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value set_data -- copy") {
        dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in, true);

        dMat sec = arma::randu<dMat>(100, 2);
        store.set_data(sec, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(sec.n_elem == out_N);
        CHECK(sec.n_rows == out_S0);
        CHECK(sec.n_cols == out_S1);
        CHECK(sec.memptr() != ptr);
        CHECK(std::abs(arma::accu(sec) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value set_data -- steal") {
        dMat in = arma::randu<dMat>(100, 2);
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(in, false);

        dMat sec = arma::randu<dMat>(200, 2);
        store.set_data(sec, false);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);

        CHECK(400 == out_N);
        CHECK(200 == out_S0);
        CHECK(2 == out_S1);
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION(" r-value set_data") {
        carma::ArrayStore<dMat> store = carma::ArrayStore<dMat>(arma::randu<dMat>(100, 2));

        store.set_data(arma::randu<dMat>(200, 2));
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(200 == out.shape(0));
        CHECK(2 == out.shape(1));
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }
} /* TEST_CASE TEST_ARRAYSTORE_MAT */

TEST_CASE("Test ArrayStore Col", "[ArrayStore<Col>]") {
    // Rows are covered by the this test case as well as they used the
    // same constructor and set_data template

    SECTION("const l-value constructor") {
        const dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value constructor -- copy") {
        dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value constructor -- steal") {
        dCol in = arma::randu<dCol>(100);
        double in_sum = arma::accu(in);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in, false);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(100 == out.shape(0));
        CHECK(std::abs(in_sum - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION(" r-value constructor") {
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(arma::randu<dCol>(100));
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(100 == out.shape(0));
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("get_view -- not writeable") {
        dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
        CHECK(carma::is_writeable(out) == false);
        CHECK_THROWS(out.mutable_unchecked());
    }

    SECTION("get_view -- not writeable") {
        dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in, true);
        py::array_t<double> out = store.get_view(true);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S0);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
        CHECK(carma::is_writeable(out) == true);
    }

    SECTION("const l-value set_data") {
        const dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in);

        const dCol sec = arma::randu<dCol>(100);
        store.set_data(sec);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(sec.n_elem == out_N);
        CHECK(sec.n_rows == out_S0);
        CHECK(sec.memptr() != ptr);
        CHECK(std::abs(arma::accu(sec) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value set_data -- copy") {
        dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in, true);

        dCol sec = arma::randu<dCol>(100);
        store.set_data(sec, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(sec.n_elem == out_N);
        CHECK(sec.n_rows == out_S0);
        CHECK(sec.memptr() != ptr);
        CHECK(std::abs(arma::accu(sec) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value set_data -- steal") {
        dCol in = arma::randu<dCol>(100);
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(in, false);

        dCol sec = arma::randu<dCol>(200);
        store.set_data(sec, false);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);

        CHECK(200 == out_N);
        CHECK(200 == out_S0);
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION(" r-value set_data") {
        carma::ArrayStore<dCol> store = carma::ArrayStore<dCol>(arma::randu<dCol>(100));

        store.set_data(arma::randu<dCol>(200));
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(200 == out.shape(0));
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }
} /* TEST_CASE TEST_ARRAYSTORE_COL */

TEST_CASE("Test ArrayStore Cube", "[ArrayStore<Cube>]") {
    SECTION("const l-value constructor") {
        const dCube in = arma::randu<dCube>(100, 2, 2);
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(in);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);
        size_t out_S2 = out.shape(2);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S1);
        CHECK(in.n_cols == out_S2);
        CHECK(in.n_slices == out_S0);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value constructor -- copy") {
        dCube in = arma::randu<dCube>(100, 2, 2);
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(in, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);
        size_t out_S2 = out.shape(2);

        CHECK(in.n_elem == out_N);
        CHECK(in.n_rows == out_S1);
        CHECK(in.n_cols == out_S2);
        CHECK(in.n_slices == out_S0);
        CHECK(in.memptr() != ptr);
        CHECK(std::abs(arma::accu(in) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value constructor -- steal") {
        dCube in = arma::randu<dCube>(100, 2, 2);
        double in_sum = arma::accu(in);
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(in, false);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(2 == out.shape(0));
        CHECK(100 == out.shape(1));
        CHECK(2 == out.shape(2));
        CHECK(std::abs(in_sum - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION(" r-value constructor") {
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(arma::randu<dCube>(100, 2, 2));
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(2 == out.shape(0));
        CHECK(100 == out.shape(1));
        CHECK(2 == out.shape(2));
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("const l-value set_data") {
        const dCube in = arma::randu<dCube>(100, 2, 2);
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(in);

        const dCube sec = arma::randu<dCube>(200, 2, 2);
        store.set_data(sec);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);
        size_t out_S2 = out.shape(2);

        CHECK(sec.n_elem == out_N);
        CHECK(sec.n_rows == out_S1);
        CHECK(sec.n_cols == out_S2);
        CHECK(sec.n_slices == out_S0);
        CHECK(sec.memptr() != ptr);
        CHECK(std::abs(arma::accu(sec) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value set_data -- copy") {
        dCube in = arma::randu<dCube>(100, 2, 2);
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(in, true);

        dCube sec = arma::randu<dCube>(200, 2, 2);
        store.set_data(sec, true);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);
        size_t out_S2 = out.shape(2);

        CHECK(sec.n_elem == out_N);
        CHECK(sec.n_rows == out_S1);
        CHECK(sec.n_cols == out_S2);
        CHECK(sec.n_slices == out_S0);
        CHECK(sec.memptr() != ptr);
        CHECK(std::abs(arma::accu(sec) - out_sum) < 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION("l-value set_data -- steal") {
        dCube in = arma::randu<dCube>(100, 2, 2);
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(in, false);

        dCube sec = arma::randu<dCube>(200, 2, 2);
        store.set_data(sec, false);
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        size_t out_N = out.size();
        size_t out_S0 = out.shape(0);
        size_t out_S1 = out.shape(1);
        size_t out_S2 = out.shape(2);

        CHECK(800 == out_N);
        CHECK(2 == out_S0);
        CHECK(200 == out_S1);
        CHECK(2 == out_S2);
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }

    SECTION(" r-value set_data") {
        carma::ArrayStore<dCube> store = carma::ArrayStore<dCube>(arma::randu<dCube>(100, 2, 2));

        store.set_data(arma::randu<dCube>(200, 2, 2));
        py::array_t<double> out = store.get_view(false);

        py::buffer_info info = out.request();
        const double* ptr = static_cast<double*>(info.ptr);

        // compute sum of array
        double out_sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(out.size()); i++) {
            out_sum += ptr[i];
        }

        CHECK(2 == out.shape(0));
        CHECK(200 == out.shape(1));
        CHECK(2 == out.shape(2));
        CHECK(std::abs(out_sum) > 1e-12);
        CHECK(carma::is_owndata(out) == false);
    }
} /* TEST_CASE TEST_ARRAYSTORE_MAT */
