#define UKF_CPP
#include "ukf.hpp"

// // template <size_t x_dim, size_t u_dim, size_t z_num, typename T>
// // Matrix<T> Ukf<x_dim, u_dim, z_num, T>::K;

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::setMeasurementFunction(Vector_f2<T> h, size_t z_dim, size_t z_idx)
{
    this->h[z_idx] = h;
    this->z_dim[z_idx] = z_dim;
    this->pre_S[z_idx].resize(z_dim, z_dim);
    this->S_inv[z_idx].resize(z_dim);
    this->h_sp[z_idx].resize(2 * x_dim, z_dim);
    this->h_sp[z_idx].fill(0);
    this->h_val[z_idx].resize(z_dim);
    this->h_val[z_idx].fill(0);
    this->Pxz[z_idx].resize(x_dim, z_dim);
}

#define max(a, b) ((a) > (b) ? (a) : (b))

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::computeStateSigmaPoints()
{
    if (f == nullptr)
        throw "Ukf::predict() state transition function not set";
    this->P.decompose();
    this->Cov_U.decompose();
    // compute state sigma points
    Vector<T> sp; // temporary sigma point
    for (size_t i = 0; i < x_dim; i++)
        this->P.D[i] = sqrt(this->P.D[i]) * sqrt_x_dim_u_dim;
    for (size_t i = 0; i < u_dim; i++)
        this->Cov_U.D[i] = sqrt(this->Cov_U.D[i]) * sqrt_x_dim_u_dim;

    internal::tmp<triangMatrix<T>> *tmp = internal::tmp<triangMatrix<T>>::get(max(x_dim, u_dim));
    // internal::tmp<Vector<T>> *sp = internal::tmp<Vector<T>>::get(x_dim); // temporary sigma point
    tmp->holdMul(this->P.L, this->P.D);
    for (size_t i = 0; i < x_dim; i++)
    {
        for (size_t j = i; j < x_dim; j++)
            this->X[j] += (*tmp)(j, i);
        sp = f(this->X, this->U, this->C);
        for (size_t j = 0; j < x_dim; j++)
            X_sp(i, j) = sp[j];
        for (size_t j = i; j < x_dim; j++)
            this->X[j] -= 2 * (*tmp)(j, i);
        sp = f(this->X, this->U, this->C);
        for (size_t j = 0; j < x_dim; j++)
            X_sp(i + x_dim, j) = sp[j];
        for (size_t j = i; j < x_dim; j++)
            this->X[j] += (*tmp)(j, i);
    }
    for (size_t i = 0; i<2*x_dim; i++)
        for (size_t j = 0; j < x_dim; j++)
            std::cout << X_sp(i, j) << " ";
    std::cout << std::endl;

    tmp->holdMul(this->Cov_U.L, this->Cov_U.D);
    for (size_t i = 0; i < u_dim; i++)
    {
        for (size_t j = i; j < u_dim; j++)
            this->U[j] += (*tmp)(j, i);
        sp = f(this->X, this->U, this->C);
        for (size_t j = 0; j < x_dim; j++)
            X_sp(i + 2 * x_dim, j) = sp[j];
        for (size_t j = i; j < u_dim; j++)
            this->U[j] -= 2 * (*tmp)(j, i);
        sp = f(this->X, this->U, this->C);
        for (size_t j = 0; j < x_dim; j++)
            X_sp(i + 2 * x_dim + u_dim, j) = sp[j];
        for (size_t j = i; j < u_dim; j++)
            this->U[j] += (*tmp)(j, i);
    }
    tmp->release();
    // sp->release();
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::computeMeasurementSigmaPoints(const size_t z_idx)
{
    if (h[z_idx] == nullptr)
        throw "Ukf::update() measurement function not set";
    this->P.decompose();
    // compute state sigma points
    // internal::tmp<Vector<T>> *sp = internal::tmp<Vector<T>>::get(z_dim[z_idx]); // temporary sigma point
    Vector<T> sp; // temporary sigma point
    for (size_t i = 0; i < x_dim; i++)
        this->P.D[i] = sqrt(this->P.D[i]) * sqrt_x_dim_u_dim;
    internal::tmp<triangMatrix<T>> *tmp = internal::tmp<triangMatrix<T>>::get(x_dim);
    tmp->holdMul(this->P.L, this->P.D);
    h_sp[z_idx].fill(-1);
    // compute measurement sigma points
    for (size_t i = 0; i < x_dim; i++)
    {
        for (size_t j = i; j < x_dim; j++)
        {
            this->X[j] += (*tmp)(j, i);
            X_sp(i, j) = X[j];
        }
        sp = h[z_idx](this->X, this->C);
        for (size_t j = 0; j < z_dim[z_idx]; j++)
            h_sp[z_idx](i, j) = sp[j];
        for (size_t j = i; j < x_dim; j++)
        {
            this->X[j] -= 2 * (*tmp)(j, i);
            X_sp(i + x_dim, j) = X[j];
        }
        sp = h[z_idx](this->X, this->C);
        for (size_t j = 0; j < z_dim[z_idx]; j++)
            h_sp[z_idx](i + x_dim, j) = sp[j];
        for (size_t j = i; j < x_dim; j++)
            this->X[j] += (*tmp)(j, i);
    }

    // sp->release();
    tmp->release();
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::predict()
{
    computeStateSigmaPoints();
    // compute predicted state (mean of sigma points)
    X.fill(0);
    for (size_t i = 0; i < X_sp.rows(); i++)
        for (size_t j = 0; j < x_dim; j++)
            X[j] += X_sp(i, j);
    X /= X_sp.rows();
    // compute predicted state covariance
    P.fill(0);
    for (size_t i = 0; i < X_sp.rows(); i++)
        for (size_t k = 0; k < x_dim; k++)
            for (size_t j = k; j < x_dim; j++)
                P(j, k) += (X_sp(i, j) - X[j]) * (X_sp(i, k) - X[k]);
    P /= X_sp.rows();
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::update(const Vector<T> &Z, const symMatrix<T> &R, const size_t z_idx)
{
    computeCrossCorrelation(z_idx);
    computeResidualPrecision(R, z_idx);
    computeResidual(Z, z_idx);
    updateStateCovariance(z_idx);
}


template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::computeCrossCorrelation(const size_t z_idx)
{
    /*
    1. Compute measurement prediction sigma points
    2. Compute the average and covariance of the sigma points
    3. Compute the cross-covariance between the state and the measurement
    4. Pre-compute the innovation covariance and store it in the pre_S matrix
    */

    // Compute measurement prediction sigma points
    computeMeasurementSigmaPoints(z_idx);

    // Compute the average and covariance of the sigma points
    // h_val[z_idx] = average(h_sp[z_idx], axis::row);
    h_val[z_idx].fill(0);
    for (size_t i = 0; i < h_sp[z_idx].rows(); i++)
        for (size_t j = 0; j < z_dim[z_idx]; j++)
            h_val[z_idx][j] += h_sp[z_idx](i, j);
    h_val[z_idx] /= h_sp[z_idx].rows();

    // Compute the cross-covariance between the state and the measurement
    Pxz[z_idx].fill(0);
    for (size_t i = 0; i < h_sp[z_idx].rows(); i++)
        for (size_t j = 0; j < x_dim; j++)
            for (size_t k = 0; k < z_dim[z_idx]; k++)
            {
                Pxz[z_idx](j, k) += (X_sp(i, j) - X[j]) * (h_sp[z_idx](i, k) - h_val[z_idx][k]);
            }
    Pxz[z_idx] /= h_sp[z_idx].rows() * 1.4142135623730951; // sqrt(2 * x_dim + 2 * u_dim)

    // Pre-compute the innovation covariance and store it in the pre_S matrix
    pre_S[z_idx].fill(0);
    for (size_t i = 0; i < h_sp[z_idx].rows(); i++)
        for (size_t j = 0; j < z_dim[z_idx]; j++)
            for (size_t k = 0; k <= j; k++)
                pre_S[z_idx](j, k) += (h_sp[z_idx](i, j) - h_val[z_idx][j]) * (h_sp[z_idx](i, k) - h_val[z_idx][k]);

    pre_S[z_idx] /= h_sp[z_idx].rows();
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::computeResidualPrecision(const symMatrix<T> &R, const size_t z_idx)
{
    S_inv[z_idx].holdAdd(pre_S[z_idx], R);
    S_inv[z_idx].holdInv(S_inv[z_idx], false);
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::computeResidual(const Vector<T> &Z, const size_t z_idx)
{
    if (z_dim[z_idx] != Z.size())
        throw "Ukf::update() measurement dimension mismatch";
    
    y[z_idx].holdSub(Z, h_val[z_idx]);
}

template <size_t x_dim, size_t u_dim, size_t c_dim, size_t z_num, typename T>
void Ukf<x_dim, u_dim, c_dim, z_num, T>::updateStateCovariance(const size_t z_idx)
{
    K.holdMul(Pxz[z_idx].T, S_inv[z_idx]);
    X.addMul(K, y[z_idx]);
    P.subMul(K, Pxz[z_idx]);
}
