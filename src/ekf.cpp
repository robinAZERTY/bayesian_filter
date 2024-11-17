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


template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
Ekf<x_dim, u_dim, c_dim, z_num, T>::Ekf()
{
    X.fill(0);
    P.fill(0);
    Cov_U.fill(0);
    dx.fill(1e-4);
    du.fill(1e-4);
    dc.fill(1e-4);
    updateMalhalanobis.fill(false);
}

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

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ekf<x_dim, u_dim, c_dim, z_num, T>::compute_S(const size_t z_idx)
{
    if (H[z_idx] == nullptr)
        this->finite_diff_H(z_idx);
    else
        H_val[z_idx] = H[z_idx](X);
    S_inv[z_idx].holdMul(*((H_val[z_idx].T) * P).release(), H_val[z_idx]);
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ekf<x_dim, u_dim, c_dim, z_num, T>::setMeasurementFunction(Vector_f2<T> h, size_t z_dim, size_t z_idx)
{
    this->h[z_idx] = h;
    this->z_dim[z_idx] = z_dim;
    this->H_val[z_idx].resize(x_dim, z_dim);
    this->S_inv[z_idx].resize(z_dim);
}

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

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
inline void Ekf<x_dim, u_dim, c_dim, z_num, T>::update(const Vector<T> &Z, const symMatrix<T> &R, const size_t z_idx)
{
    if (h[z_idx] == nullptr)
        throw "Ekf::update() measurement function not set";
    if (z_dim[z_idx] != Z.size())
        throw "Ekf::update() measurement dimension mismatch";

    h_val[z_idx] = h[z_idx](X, C); //  measurement prediction

    // measurement function Jacobian computation
    if (H[z_idx] == nullptr)
        this->finite_diff_H(z_idx);
    else
        H_val[z_idx] = H[z_idx](X, C);

    H_P[z_idx].holdMul(H_val[z_idx].T, P); // common expression (calculus factorisation)
    y[z_idx].holdSub(Z, h_val[z_idx]);     // residual
    
    // residual precision
    S_inv[z_idx].holdMul(H_P[z_idx], H_val[z_idx]);
    S_inv[z_idx] += R;
    S_inv[z_idx].holdInv(S_inv[z_idx], false);

    K.holdMul(H_P[z_idx].T, S_inv[z_idx]); // Kalman gains
    X.addMul(K, y[z_idx]);             // update the state
    P.subMul(K, H_P[z_idx]);           // update the state covariance

    if (!updateMalhalanobis[z_idx])
        return;
        
    T dd = y[z_idx].dot(*(S_inv[z_idx] * y[z_idx]).release()); // squared Mahalanobis distance
    dd /= z_dim[z_idx];                                    // average it over dimensions (should be around 1)
    ds[z_idx] += alpha * (dd - ds[z_idx]);                 // low pass filter (or exponential moving average)
}