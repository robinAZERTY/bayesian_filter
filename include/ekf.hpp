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

#ifndef EKF_HPP
#define EKF_HPP

#include "matrix.hpp"
#include "symMatrix.hpp"
using namespace operators;

/**
 * @class Ekf
 * @brief A generic implementation of an Extended Kalman Filter (EKF).
 *
 * This class provides the core EKF functionality, including state prediction and update steps for nonlinear systems.
 * It supports dynamic system models and allows for numerical differentiation of Jacobians for both state transition and measurement functions.
 *
 * You must provide several key functions and data to ensure the filter works properly:
 *
 * - **State Transition Function** (`f`): A function that models the evolution of the state vector over time, used for the prediction step.
 * - **Measurement Function** (`h`): A function that models the measurement process, mapping the state vector to the measurement space.
 * - **Initial State and Covariance** (`X` and `P`): The user must initialize the state vector `X` and the state covariance matrix `P` with reasonable values for the system.
 * - **Control Inputs** (`U`): If the system involves control inputs, the user must provide them during the prediction step.
 * - **Measurement Data** (`Z` and `R`): During the update step, the user must provide the measurement vector `Z` and the measurement noise covariance `R`.
 *
 * The EKF will then handle the state prediction, measurement update, and state covariance estimation.
 *
 * The filter supports multiple types of sensors, with each sensor having its own measurement function and Jacobian.
 * The number of sensor types (`z_num`) can be customized, allowing for flexibility in handling various measurements within the same filter.
 *
 * The class also supports numerical differentiation for computing Jacobians if analytical expressions are not available, making it versatile for different models.
 *
 * @tparam x_dim The dimension of the state vector.
 * @tparam u_dim The dimension of the control input vector.
 * @tparam c_dim The dimension of the system parameters (default is 1).
 * @tparam z_num The number of measurement types (default is 1).
 * @tparam T The data type used for calculations (default is float).
 */
template <size_t x_dim, size_t u_dim, size_t c_dim = 1, size_t z_num = 1, typename T = float>
class Ekf
{
#ifdef EVERYTHING_PUBLIC
public:
#else
private:
#endif

    /** State transition function */
    Vector_f3<T> f = nullptr;

    /** Measurement dimensions for each measurement type */
    size_t z_dim[z_num] = {0};

    /** Measurement functions for each measurement type */
    Vector_f2<T> h[z_num] = {nullptr};

    /** State transition Jacobians */
    Matrix_f3<T> Fx = nullptr;
    Matrix_f3<T> Fu = nullptr;
    Matrix_f3<T> Fc = nullptr;

    /** Measurement Jacobians for each measurement type */
    Matrix_f2<T> H[z_num] = {nullptr};

    /** Measurement residuals for each measurement type */
    Vector<T> y[z_num];

    /** Jacobian values for measurement functions */
    Matrix<T> H_val[z_num];

    /** partial computation of innovation covariance (without R added) */
    symMatrix<T> pre_S[z_num];

    /** innovation precision */
    ldl_matrix<T> S_inv[z_num];

    /** Predicted measurement for each measurement type */
    Vector<T> h_val[z_num];

    /** Previous state vector */
    Vector<T> prev_X = Vector<T>(x_dim);

    /** State transition matrices */
    Matrix<T> Fx_val_T = Matrix<T>(x_dim, x_dim);
    Matrix<T> Fu_val_T = Matrix<T>(u_dim, x_dim);
    Matrix<T> Fc_val_T = Matrix<T>(c_dim, x_dim);

    /** Kalman gain */
    rowMajorMatrix<T> K;

    /** Measurement Jacobian * State covariance for each measurement type */
    Matrix<T> H_P[z_num];

    inline void finite_diff_Fx(const size_t i);
    inline void finite_diff_Fu(const size_t i);

    /**
     * @brief Numerical differentiation for the state transition Jacobian (Fx) using finite differences.
     *
     * This function approximates the Jacobian of the state transition function with respect to the state vector.
     *
     * @param i The index of the state dimension to compute the Jacobian.
     */
    void finite_diff_Fx()
    {
        for (size_t i = 0; i < x_dim; i++)
            finite_diff_Fx(i);
    };

    /**
     * @brief Numerical differentiation for the control input transition Jacobian (Fu) using finite differences.
     *
     * This function approximates the Jacobian of the state transition function with respect to the control input vector.
     *
     * @param i The index of the control input dimension to compute the Jacobian.
     */
    void finite_diff_Fu()
    {
        for (size_t i = 0; i < u_dim; i++)
            finite_diff_Fu(i);
    };

    /**
     * @brief Numerical differentiation for the measurement Jacobian (H) using finite differences.
     *
     * This function approximates the Jacobian of the measurement function with respect to the state vector.
     *
     * @param z_idx The index of the measurement type.
     * @param i The index of the state dimension to compute the Jacobian.
     */
    void finite_diff_H(const size_t z_idx, const size_t i);

    /** Compute Jacobian for measurements using numerical differentiation */
    void finite_diff_H(const size_t z_idx)
    {
        for (size_t i = 0; i < x_dim; i++)
            finite_diff_H(z_idx, i);
    };

    /** Flag indicating whether the filter is initialized */
    bool initted = false;

public:
    /** Current state vector */
    Vector<T> X = Vector<T>(x_dim);

    /** Small state epsilon for numerical differentiation */
    Vector<T> dx = Vector<T>(x_dim);

    /** State covariance matrix */
    symMatrix<T> P = symMatrix<T>(x_dim, x_dim);

    /** Control input vector */
    Vector<T> U = Vector<T>(u_dim);

    /** Small control input epsilon for numerical differentiation */
    Vector<T> du = Vector<T>(u_dim);

    /** Process noise covariance matrix */
    symMatrix<T> Cov_U = symMatrix<T>(u_dim);

    /** System parameters vector */
    Vector<T> C = Vector<T>(c_dim);

    /** Small system parameters epsilon for numerical differentiation */
    Vector<T> dc = Vector<T>(c_dim);

    /** Flags indicating whether to update the Mahalanobis distance for each measurement type */
    Vector<bool> updateMahalanobis = Vector<bool>(z_num);

    /**
     * @brief Default constructor for the EKF filter.
     *
     * Initializes the state vector `X`, the state covariance `P`, and other necessary vectors to default values.
     * Sets small values for numerical differentiation and initializes Mahalanobis distance tracking.
     */
    Ekf();

    /** Constructor with a state transition function */
    Ekf(Vector_f3<T> f) : Ekf() { setPredictionFunction(f); }

    /** Destructor */
    ~Ekf() {};

    /** Set the state prediction function */
    void setPredictionFunction(Vector_f3<T> f) { this->f = f; }

    /** Set the Jacobian for state transition */
    void setJacobianFunction_Fx(Matrix_f3<T> Fx) { this->Fx = Fx; }

    /** Set the Jacobian for control input transition */
    void setJacobianFunction_Fu(Matrix_f3<T> Fu) { this->Fu = Fu; }

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
    void setMeasurementFunction(Vector_f2<T> h, size_t z_dim, size_t z_idx = 0);

    /** Set the Jacobian for the measurement function */
    void setJacobianFunction_H(Matrix_f2<T> H, size_t z_idx = 0) { this->H[z_idx] = H; }

    /**
     * @brief Predicts the state using the state transition function.
     *
     * This function computes the predicted state `X` based on the previous state and the state transition function.
     * It also updates the state covariance matrix `P` based on the state transition Jacobian.
     *
     * Throws an exception if the state transition function is not set.
     */
    inline void predict();

    /**
     * @brief First part of the update function, computes the measurement prediction and the measurement Jacobian.
     *
     * Additionally, it computes the common expression `H*P` which is used to reduce the number of matrix multiplications during the update step.
     *
     * @param z_idx The index of the measurement type.
     */
    inline void compute_H_P(const size_t z_idx = 0);

    /**
     * @brief Second part of the update function, computes the residual and the residual precision.
     *
     * This function is a part of the update step.
     *
     * @param Z The new measurement vector.
     * @param R The measurement noise covariance matrix.
     * @param z_idx The index of the measurement type.
     */
    inline void computeResidualPrecision(const symMatrix<T> &R, const size_t z_idx = 0);

    /**
     * @brief Third part of the update function, computes the residual and updates the state and covariance.
     *
     *  This function compute the difference between the measurement and the predicted measurement
     *
     * @param Z The new measurement vector.
     * @param z_idx The index of the measurement type.
     */
    inline void computeResidual(const Vector<T> &Z, const size_t z_idx = 0);

    /**
     * @brief Final part of the update function, computes the state and covariance.
     *
     * This function compute the Kalman gain and updates the state and covariance based on the measurement residual and the residual precision.
     *
     * @param z_idx The index of the measurement type.
     * @param updateMahalanobis Flag indicating whether to update the Mahalanobis distance.
     */
    inline void updateStateCovariance(const size_t z_idx = 0);

    /**
     * @brief Compute the Mahalanobis distance for a specific measurement type relative to last update of residual and residual precision.
     *
     * Should be called after the "computeResidualPrecision" function
     * Can be used to monitor the likelihood of the given measurement.
     * It is useful for detecting outliers or resolving identification issues (when you receive a measurement and you don't know witch measurement type it is).
     * If everything is perfectly modelled, the average Mahalanobis distance should be 1 (after averaging over a long period).
     *
     * @param z_idx The index of the measurement type.
     * @return The Mahalanobis distance.
     */
    inline T mahalanobisDistance(const size_t z_idx = 0) { return y[z_idx].dot(*(S_inv[z_idx] * y[z_idx]).release()); }

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
    inline void update(const Vector<T> &Z, const symMatrix<T> &R, const size_t z_idx = 0);
};

#ifndef EKF_CPP
#include "ekf.cpp"
#endif

#endif // EKF_HPP