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

#define EKF_CPP
#include "ekf.hpp"



/**
 * @brief Default constructor for the EKF filter.
 * 
 * Initializes the state vector `X`, the state covariance `P`, and other necessary vectors to default values. 
 * Sets small values for numerical differentiation and initializes Mahalanobis distance tracking.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
Ekf<x_dim, u_dim, c_dim, z_num, T>::Ekf()
{
    X.fill(0);
    P.fill(0);
    Cov_U.fill(0);
    dx.fill(1e-4);
    du.fill(1e-4);
    dc.fill(1e-4);
    updateMahalanobis.fill(false);
}

/**
 * @brief Numerical differentiation for the state transition Jacobian (Fx) using finite differences.
 * 
 * This function approximates the Jacobian of the state transition function with respect to the state vector.
 * 
 * @param i The index of the state dimension to compute the Jacobian.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::finite_diff_Fx(const size_t i)
{
    const T eps = dx[i];
    prev_X[i] += eps;
    Vector<T> dX(&Fx_val_T[i * x_dim], x_dim, true);
    dX.holdSub(*f(prev_X, U, C).release(), X);
    dX /= eps;
    prev_X[i] -= eps;
}

/**
 * @brief Numerical differentiation for the control input transition Jacobian (Fu) using finite differences.
 * 
 * This function approximates the Jacobian of the state transition function with respect to the control input vector.
 * 
 * @param i The index of the control input dimension to compute the Jacobian.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::finite_diff_Fu(const size_t i)
{
    const T eps = du[i];
    U[i] += eps;
    Vector<T> dX(&Fu_val_T[i * x_dim], x_dim, true);
    dX.holdSub(*f(prev_X, U, C).release(), X);
    dX /= eps;
    U[i] -= eps;
}

/**
 * @brief Numerical differentiation for the measurement Jacobian (H) using finite differences.
 * 
 * This function approximates the Jacobian of the measurement function with respect to the state vector.
 * 
 * @param z_idx The index of the measurement type.
 * @param i The index of the state dimension to compute the Jacobian.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::finite_diff_H(const size_t z_idx, const size_t i)
{
    const T eps = dx[i];
    X[i] += eps;
    Vector<T> dz(&H_val[z_idx][i * z_dim[z_idx]], z_dim[z_idx], true);
    dz.holdSub(*h[z_idx](X, C).release(), h_val[z_idx]);
    dz /= eps;
    X[i] -= eps;
}

/**
 * @brief Computes the innovation covariance (S) for the specified measurement type.
 * 
 * This function calculates the inverse of the innovation covariance matrix using the measurement Jacobian and the state covariance.
 * 
 * @param z_idx The index of the measurement type.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ekf<x_dim, u_dim, c_dim, z_num, T>::compute_S(const size_t z_idx)
{
    if (H[z_idx] == nullptr)
        this->finite_diff_H(z_idx);
    else
        H_val[z_idx] = H[z_idx](X);
    S_inv[z_idx].holdMul(*((H_val[z_idx].T) * P).release(), H_val[z_idx]);
}

/**
 * @brief Sets the measurement function for a specific sensor type.
 * 
 * This function defines the measurement function and measurement dimension for a particular sensor type.
 * It also initializes the corresponding Jacobians and matrices.
 * 
 * @param h The measurement function for the sensor type.
 * @param z_dim The dimension of the measurement vector for the sensor.
 * @param z_idx The index of the sensor type (default is 0).
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ekf<x_dim, u_dim, c_dim, z_num, T>::setMeasurementFunction(Vector_f2<T> h, size_t z_dim, size_t z_idx)
{
    this->h[z_idx] = h;
    this->z_dim[z_idx] = z_dim;
    this->H_val[z_idx].resize(x_dim, z_dim);
    this->S_inv[z_idx].resize(z_dim);
}

/**
 * @brief Predicts the state using the state transition function.
 * 
 * This function computes the predicted state `X` based on the previous state and the state transition function.
 * It also updates the state covariance matrix `P` based on the state transition Jacobian.
 * 
 * Throws an exception if the state transition function is not set.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::predict()
{

    if (f == nullptr)
        throw "Ekf::predict() state transition function not set";

    if (!initted)
    {
        prev_X = X;
        initted = true;
        C[0] = 0;
    }
    else
        prev_X = X;

    X = f(prev_X, U, C);

    if (Fx != nullptr)
        Fx_val_T = Fx(prev_X, U, C);
    else
        this->finite_diff_Fx();
    if (Fu != nullptr)
        Fu_val_T = Fu(prev_X, U, C);
    else
        this->finite_diff_Fu();

    P.holdMul(*(Fx_val_T.T * P).release(), Fx_val_T);
    P.addMul(*(Fu_val_T.T * Cov_U).release(), Fu_val_T);
}

/**
 * @brief Updates the state and covariance matrix using a new measurement and measurement noise covariance.
 * 
 * This function updates the state vector `X` and the state covariance matrix `P` based on the new measurement `Z`
 * and the measurement noise covariance matrix `R`. The Kalman gain is also computed and used to adjust the state.
 * Throws an exception if the measurement function is not set or if there is a dimension mismatch.
 * This function can be split into multiple parts for better readability and modularity as follows:
 * @code
 * compute_H_P(z_idx);
 * computeResidualPrecision(R, z_idx);
 * computeResidual(Z, z_idx);
 * updateStateCovariance(z_idx);
 * @endcode
 * Before calling @code updateStateCovariance @endcode, you can check the Mahalanobis distance using @ref mahalanobisDistance.
 * 
 * @param Z The new measurement vector.
 * @param R The measurement noise covariance matrix.
 * @param z_idx The index of the measurement type (default is 0).
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::update(const Vector<T> &Z, const symMatrix<T> &R, const size_t z_idx)
{
    compute_H_P(z_idx);
    computeResidualPrecision(R, z_idx);
    computeResidual(Z, z_idx);
    updateStateCovariance(z_idx);
}

/**
 * @brief First part of the update function, computes the measurement prediction and the measurement Jacobian.
 * 
 * Additionally, it computes the common expression `H*P` which is used to reduce the number of matrix multiplications during the update step.
 * 
 * @param z_idx The index of the measurement type.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ekf<x_dim, u_dim, c_dim, z_num, T>::compute_H_P(const size_t z_idx)
{
    if (h[z_idx] == nullptr)
        throw "Ekf::update() measurement function not set";

    h_val[z_idx] = h[z_idx](X, C); //  measurement prediction

    // measurement function Jacobian computation
    if (H[z_idx] == nullptr)
        this->finite_diff_H(z_idx);
    else
        H_val[z_idx] = H[z_idx](X, C);

    H_P[z_idx].holdMul(H_val[z_idx].T, P); // common expression (calculus factorisation)
    pre_S[z_idx].holdMul(H_P[z_idx], H_val[z_idx]); // pre computation of residual covariance (R not added yet)
}


/**
 * @brief Second part of the update function, computes the residual and the residual precision.
 * 
 * This function is a part of the update step.
 * 
 * @param Z The new measurement vector.
 * @param R The measurement noise covariance matrix.
 * @param z_idx The index of the measurement type.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::computeResidualPrecision(const symMatrix<T> &R, const size_t z_idx)
{    
    // residual precision
    S_inv[z_idx].holdAdd(pre_S[z_idx], R);
    S_inv[z_idx].holdInv(S_inv[z_idx], false);
}

/**
 * @brief Third part of the update function, computes the residual and updates the state and covariance.
 * 
 *  This function compute the difference between the measurement and the predicted measurement
 * 
 * @param Z The new measurement vector.
 * @param z_idx The index of the measurement type.
 */
template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::computeResidual(const Vector<T> &Z, const size_t z_idx)
{
    if (z_dim[z_idx] != Z.size())
        throw "Ekf::update() measurement dimension mismatch";

    y[z_idx].holdSub(Z, h_val[z_idx]);
}


/**
 * @brief Final part of the update function, computes the state and covariance.
 * 
 * This function compute the Kalman gain and updates the state and covariance based on the measurement residual and the residual precision.
 * 
 * @param z_idx The index of the measurement type.
 * @param updateMahalanobis Flag indicating whether to update the Mahalanobis distance.
 */

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::updateStateCovariance(const size_t z_idx)
{
    K.holdMul(H_P[z_idx].T, S_inv[z_idx]); // Kalman gains
    X.addMul(K, y[z_idx]);             // update the state
    P.subMul(K, H_P[z_idx]);           // update the state covariance
}