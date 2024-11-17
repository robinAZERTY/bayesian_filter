/*
 * Copyright (C) 2024 robinAZERTY [https://github.com/robinAZERTY]
 *
 * This file is part of ESP32AlgebraFilters library.
 *
 * ESP32AlgebraFilters library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESP32AlgebraFilters library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ESP32AlgebraFilters library. If not, see <https://www.gnu.org/licenses/>.
 */

/*
The Ekf class implements the Extended Kalman Filter (EKF), a recursive filter used for estimating the state of a dynamic system from noisy measurements. 
This updated version introduces a more streamlined and flexible approach to EKF implementation compared to the previous version.

Key changes:
- Improved modularity: The class now uses templates and function pointers for greater flexibility in defining state transition and measurement functions.
- Reduced memory usage: Temporary matrices and vectors are allocated dynamically as needed, optimizing memory consumption.
- Simplified interface: The class provides template-based methods for prediction, measurement, and update steps, allowing for easy customization and extension.

functional methods:
- predict(): Propagates the state using the provided state transition function and command input, along with the specified process noise covariance matrix.
- update(): Updates the state based on the measurement, using the predicted measurement, its associated innovation covariance matrix, and optionally, the measurement Jacobian

Next steps:
- Integration with other filtering techniques such as the Unscented Kalman Filter (UKF) for comparison and performance evaluation.
*/

#ifndef EKF_HPP
#define EKF_HPP

#include "matrix.hpp"
#include "symMatrix.hpp"
using namespace operators;

template <size_t x_dim, size_t u_dim, size_t c_dim = 1, size_t z_num = 1, typename T = float>
class Ekf
{
    #ifdef EVERYTHING_PUBLIC
    public:
    #else
    private:
    #endif
    
    Vector_f3<T> f = nullptr;          // state transition function
    size_t z_dim[z_num] = {0};         // measurement dimensions
    Vector_f2<T> h[z_num] = {nullptr}; // measurement functions
    Matrix_f3<T> Fx = nullptr;         // state transition Jacobian
    Matrix_f3<T> Fu = nullptr;         // state transition Jacobian
    Matrix_f3<T> Fc = nullptr;         // state transition Jacobian
    Matrix_f2<T> H[z_num] = {nullptr}; // measurement Jacobian
    Vector<T> y[z_num];                     // measurement residuals
    float ds[z_num]={1};                     // measurement squared Mahalanobis distance with the estimated state
    float alpha = 1-5.f/(5+1);                       // filter forgetting factor

    Matrix<T> H_val[z_num];                 // measurement Jacobian values
    ldl_matrix<T> S_inv[z_num];                 // innovation covariance
    Vector<T> h_val[z_num];                     // predicted measurement
    Vector<T> prev_X = Vector<T>(x_dim);        // previous state
    Matrix<T> Fx_val_T = Matrix<T>(x_dim, x_dim); // state transition matrices
    Matrix<T> Fu_val_T = Matrix<T>(u_dim, x_dim); // state transition matrices
    Matrix<T> Fc_val_T = Matrix<T>(c_dim, x_dim); // state transition matrices
    rowMajorMatrix<T> K;

    Matrix<T> H_P[z_num];
    inline void finite_diff_Fx(const size_t i);
    inline void finite_diff_Fu(const size_t i);
    void finite_diff_Fx(){for (size_t i = 0; i < x_dim; i++) finite_diff_Fx(i);};
    void finite_diff_Fu(){for (size_t i = 0; i < u_dim; i++) finite_diff_Fu(i);};

    void finite_diff_H(const size_t z_idx, const size_t i);
    void finite_diff_H(const size_t z_idx){for (size_t i = 0; i < x_dim; i++) finite_diff_H(z_idx, i);};
    void compute_S(const size_t z_idx);

    bool initted = false;
public:
    Vector<T> X = Vector<T>(x_dim);              // state
    Vector<T> dx = Vector<T>(x_dim);             // small state epsilon for numerical differentiation
    symMatrix<T> P = symMatrix<T>(x_dim, x_dim); // state covariance
    Vector<T> U = Vector<T>(u_dim);              // command input
    Vector<T> du = Vector<T>(u_dim);             // small command epsilon for numerical differentiation
    symMatrix<T> Cov_U = symMatrix<T>(u_dim);        // process noise covariance
    Vector<T> C = Vector<T>(c_dim);                  // system parameters
    Vector<T> dc = Vector<T>(c_dim);                 // small system parameters epsilon for numerical differentiation
    Vector<bool> updateMahalanobis = Vector<bool>(z_num); // update the Mahalanobis distance
    Ekf();
    Ekf(Vector_f3<T> f) : Ekf() { setPredictionFunction(f); }
    ~Ekf(){};

    void setPredictionFunction(Vector_f3<T> f) { this->f = f; }
    void setJacobianFunction_Fx(Matrix_f3<T> Fx) { this->Fx = Fx; }
    void setJacobianFunction_Fu(Matrix_f3<T> Fu) { this->Fu = Fu; }

    void setMeasurementFunction(Vector_f2<T> h, size_t z_dim, size_t z_idx = 0);
    void setJacobianFunction_H(Matrix_f2<T> H, size_t z_idx = 0) { this->H[z_idx] = H; }

    inline void predict();

    inline void update(const Vector<T> &Z, const symMatrix<T> &R, const size_t z_idx = 0);

    inline float getMahalanobisDistance(const size_t z_idx = 0) const { return ds[z_idx]; }
};


#ifndef EKF_CPP
#include "ekf.cpp"
#endif

#endif // EKF_HPP