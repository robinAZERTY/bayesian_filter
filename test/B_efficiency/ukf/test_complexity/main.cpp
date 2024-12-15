/*
to run all the test use the following command
pio test -e native
*/

#define EVERYTHING_PUBLIC
#include <ukf.hpp>
#include "watcher.hpp"
#include <unity.h>

void setUp() {
    // Initialisation avant chaque test (laisser vide si inutile)
}

void tearDown() {
    // Nettoyage après chaque test (laisser vide si inutile)
}


template <typename T>
internal::tmp<Vector<T>> && f(const Vector<T> &x, const Vector<T> &u, const Vector<T> &c)
{
    auto *ret = internal::tmp<Vector<T>>::get(x.size());
    ret->hold(x);
    return internal::move(*ret);
}

template <typename T, size_t z_dim>
internal::tmp<Vector<T>> && h(const Vector<T> &x, const Vector<T> &c)
{
    auto *ret = internal::tmp<Vector<T>>::get(z_dim);
    (*ret).fill(0);
    return internal::move(*ret);
}

Watcher w3("prediction");
Watcher w4("update");


template <size_t X_DIM, size_t U_DIM, size_t Z_DIM>
void ukfSpeedTest(const size_t N = 10)
{
  
  // internal::tmp<Vector<float>>::freeAll();
  // internal::tmp<rowMajorMatrix<float>>::freeAll();  
  // internal::tmp<Matrix<float>>::freeAll();
  internal::tmp<triangMatrix<float>>::freeAll();
  for (size_t i = 0; i < N; i++)
  {
    Watcher::resetAll();
    Ukf<X_DIM, U_DIM> ukf(f<float>);
    ukf.setMeasurementFunction(h<float,Z_DIM>, Z_DIM);
    symMatrix<float> R(Z_DIM);
    Vector<float> Z(Z_DIM);
    ukf.X.fill(0);
    ukf.P.fill(0);
    ukf.Cov_U.fill(0);
    R.fill(0);
    w3.start();
    ukf.U.fill(0);
    ukf.predict();
    w3.stop();
    w4.start();
    Z.fill(0);
    ukf.update(Z, R);
    w4.stop();
    Serial.print(X_DIM);
    Serial.print("\t");
    Serial.print(U_DIM);
    Serial.print("\t");
    Serial.print(Z_DIM);
    Serial.print("\t");
    Serial.print(w3.get());
    Serial.print("\t");
    Serial.print(w4.get());
    Serial.print("\t");
    Serial.print(internal::alloc_count + sizeof(ukf));
    Serial.println();
  }
}
#define eq_xxx(x) ((x*x*x)<30000)

#define eq_xxu(x,u) ((x*x*u)<10000)
#define eq_xux(x,u) ((x*u*x)<8000)
#define eq_xxz(x,z) ((x*x*z)<10000)
#define eq_xzz(x,z) ((x*z*z)<4000)
#define eq_zzz(z) ((z*z*z)<40000)

#define eq(x,u,z) ((x+u+z)<60)
#define eq2(x,u,z) (eq(x,u,z) && eq_xxx(x) && eq_xxu(x,u) && eq_xux(x,u) && eq_xxz(x,z) && eq_xzz(x,z) && eq_zzz(z))
#define step 3

template <size_t x, size_t u, size_t z>
void IncrementalTest_z(size_t n = 1)
{
  if constexpr (eq2(x,u,z))
  {
    ukfSpeedTest<x,u,z>(n);
    IncrementalTest_z<x,u,z+step>(n);
  }
}

template <size_t x, size_t u, size_t z>
void IncrementalTest_uz(size_t n = 1)
{
  if constexpr (eq2(x,u,z))
  {
    IncrementalTest_z<x,u,z>(n);
    IncrementalTest_uz<x,u+step,z>(n);
  }
}

template <size_t x, size_t u, size_t z>
void IncrementalTest_xuz(size_t n = 1)
{
  if constexpr (eq2(x,u,z))
  {
    IncrementalTest_uz<x,u,z>(n);
    IncrementalTest_xuz<x+step,u,z>(n);
  }
}

void setup() {
    Serial.begin(115200);
    delay(200); // Permet d'attendre que la liaison série soit établie
    Serial.println("ukf-go");// send ukf-go to notify the start of the test
    Serial.println("X_DIM\tU_DIM\tZ_DIM\tprediction\tupdate\talloc");
    IncrementalTest_xuz<1,1,1>(5);
    Serial.println("end");// send esp-end to notify the end of the test

    UNITY_BEGIN();
    TEST_ASSERT_TRUE(true);
    UNITY_END();
}

void loop() {
}
