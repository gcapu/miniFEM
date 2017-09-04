#include "gtest/gtest.h"

#include <chrono>
#include <Eigen/Geometry>

#include "minifem.h"


template<typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA>& a,
  const Eigen::DenseBase<DerivedB>& b,
  const typename DerivedA::RealScalar& rtol
  = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
  const typename DerivedA::RealScalar& atol
  = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
  {
  return ((a.derived() - b.derived()).array().abs()
    <= (atol + rtol * b.derived().array().abs())).all();
  }

TEST(isolinear, exists)
  {
  typedef float scalar;
  Eigen::Matrix<scalar, 3,3> E = Eigen::Matrix<scalar, 3, 3>::Random()/10.;
  E.triangularView<Eigen::StrictlyLower>() = E.triangularView<Eigen::StrictlyUpper>().transpose();
  Eigen::Matrix<scalar,6,6> K;
  scalar el = 200;
  scalar nu = 0.3;
  scalar g = .5*el / (1 + nu);
  scalar ee = el / ((1. + nu) * (1. - 2. * nu));
  K << (1-nu)*ee, nu*ee    , nu*ee    , 0, 0, 0,
       nu*ee    , (1-nu)*ee, nu*ee    , 0, 0, 0,
       nu*ee    , nu*ee    , (1-nu)*ee, 0, 0, 0,
       0        , 0        , 0        , g, 0, 0,
       0        , 0        , 0        , 0, g, 0,
       0        , 0        , 0        , 0, 0, g;
  Eigen::Matrix<scalar,3,3> S;
  Eigen::Matrix<scalar,6,1> ev;
  ev << E(0,0), E(1,1), E(2,2), E(0,1), E(0,2), E(1,2);
  Eigen::Matrix<scalar,6,1> sv = K*ev;
  S << sv(0), sv(3), sv(4),
       sv(3), sv(1), sv(5),
       sv(4), sv(5), sv(2);
  mini::IsotropicLinear<scalar, 3> mat(el,nu);
  EXPECT_TRUE(allclose(mat.Stiffness(), K));
  EXPECT_TRUE(allclose(mat.Stress(E), S));
  }

