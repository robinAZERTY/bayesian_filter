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

#ifndef UKF_HPP
#define UKF_HPP
#include "matrix.hpp"
#include "symMatrix.hpp"
using namespace operators;


template <size_t x_dim, size_t u_dim, size_t c_dim = 1, size_t z_num = 1, typename T = float>
class Ukf
{
private:

    /** State transition function */
    Vector_f3<T> f = nullptr;

    /** Measurement dimensions for each measurement type */
    size_t z_dim[z_num] = {0};

    /** Measurement functions for each measurement type */
    Vector_f2<T> h[z_num] = {nullptr};

    /** Measurement residuals for each measurement type */
    Vector<T> y[z_num];

    /** partial computation of innovation covariance (without R added) */
    symMatrix<T> pre_S[z_num];

    /** innovation precision */
    ldl_matrix<T> S_inv[z_num];

    /** Predicted measurement for each measurement type */
    Vector<T> h_val[z_num];

    /** Previous state vector */
    Vector<T> prev_X = Vector<T>(x_dim);

    /** Kalman gain */
    rowMajorMatrix<T> K;

    /** State sigma points */
    Matrix<T> X_sp = Matrix<T>(2 * x_dim + 2 * u_dim, x_dim);

    /** Measurement sigma points */
    Matrix<T> h_sp[z_num];

    /** State-Measurement cross-covariance */
    Matrix<T> Pxz[z_num];

    const T sqrt_x_dim_u_dim = sqrt(x_dim + u_dim);
    const T sqrt_x_dim = sqrt(x_dim);
    void computeStateSigmaPoints();
    void computeMeasurementSigmaPoints(const size_t z_idx);

    /** Flag indicating whether the filter is initialized */
    bool initted = false;

public:
    /** Current state vector */
    Vector<T> X = Vector<T>(x_dim);

    /** State covariance matrix */
    ldl_matrix<T> P = ldl_matrix<T>(x_dim);

    /** Control input vector */
    Vector<T> U = Vector<T>(u_dim);

    /** Process noise covariance matrix */
    ldl_matrix<T> Cov_U = ldl_matrix<T>(u_dim);

    /** System parameters vector */
    Vector<T> C = Vector<T>(c_dim);

    /**
     * @brief Default constructor for the EKF filter.
     *
     * Initializes the state vector `X`, the state covariance `P`, and other necessary vectors to default values.
     * Sets small values for numerical differentiation.
     */
    Ukf(){};

    /** Constructor with a state transition function */
    Ukf(Vector_f3<T> f) : Ukf() { setPredictionFunction(f); }

    /** Destructor */
    ~Ukf() {};

    /** Set the state prediction function */
    void setPredictionFunction(Vector_f3<T> f) { this->f = f; }

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

    /**
     * @brief Predicts the state using the state transition function.
     *
     * This function computes the predicted state `X` based on the previous state and the state transition function.
     * It also updates the state covariance matrix `P` based on the state sigma points.
     *
     * Throws an exception if the state transition function is not set.
     */
    inline void predict();

    /**
     * @brief First part of the update function, computes cross-correlation between state and measurement.
     *
     * By computing the measurement prediction sigma points, it get the expected measurement value and measurement covariance.
     * Then, it use the last state sigma points generated from the last prediction to compute the cross-covariance between the state and the measurement.
     * 
     * Additionally, it pre-computes the innovation covariance and stores it in the pre_S matrix, which helps to reduce redundant computations.
     *
     * @param z_idx The index of the measurement type.
     */
    inline void computeCrossCorrelation(const size_t z_idx = 0);

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

#ifndef UKF_CPP
#include "ukf.cpp"
#endif

#endif // UKF_HPP