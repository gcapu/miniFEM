#include "gtest/gtest.h"

#include "minifem.h"

// Helper 
template<typename DerivedA, typename DerivedB>
bool allclose2(const Eigen::DenseBase<DerivedA>& a,
    const Eigen::DenseBase<DerivedB>& b,
    const typename DerivedA::RealScalar& rtol
    = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar& atol
    = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
  {
  return ((a.derived() - b.derived()).array().abs()
    <= (atol + rtol * b.derived().array().abs())).all();
  }


//----------------------------------------------------------------------------
// FEM
//----------------------------------------------------------------------------

TEST(miniFEM, patchC3D20Rdouble)
{
  mini::FEM<double, 3> testModel;
  double E = 5, nu = 0.3;
  mini::IsotropicLinear<double, 3> isoMat(E, nu);
  bool readSuccess = testModel.ReadAbaqusInp("../benchmark/input/patchC3D20R.inp", isoMat);
  ASSERT_TRUE(readSuccess);
  Eigen::VectorXd zerovec = Eigen::VectorXd::Zero(testModel.F().size());
  //there should be zero forces initially
  EXPECT_TRUE(allclose2(testModel.F(), zerovec));
  //stretching in the y direction
  double d = 1;
  for(int i = 0; i<testModel.NumNodes(); i++)
    {
    mini::FEM<double, 3>::NodeType::VectorType U;
    U << 0 , d*testModel.Node(i).X(1), 0;
    testModel.setuNode(i, U);
    }
  //getting the total force on two of the faces
  double fx = 0, fy = 0;
  Eigen::MatrixXd f = testModel.F();
  for(int i = 0; i<testModel.NumNodes(); i++){
    if(testModel.Node(i).X(0) > 0.9)
      fx += f(i*3);
    if (testModel.Node(i).X(1) > 0.9)
      fy += f(i*3+1);
  }
  //Using plane strain we can get the correct result. 
  double E11 = .5*((1.+d)*(1.+d)-1.)/1.;
  double S11 = E11 * E*(1-nu)/(1+nu)/(1-2*nu);
  double S22 = E11 * E* nu/(1+nu)/(1-2*nu);
  double area = 1.;
  EXPECT_NEAR(fy, area*S11*(d+1.)/1., 1e-10);
  EXPECT_NEAR(fx, area*S22, 1e-10);
  
  /* disabled temporarily. I have to check if it is true for this element
  //rotating the entire model should give the same forces
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(2*atan(1), Eigen::Vector3d::UnitZ());
  for(int i = 0; i<testModel.NumNodes(); i++)
    {
    mini::FEM<double, 3>::Node::VectorType U;
    U = R * testModel.getNode(i).x() - testModel.getNode(i).X();
    testModel.setuNode(i, U);
    }
  //however they are in different directions now
  fx = 0; fy = 0;
  f = testModel.F();
  for(int i = 0; i<testModel.NumNodes(); i++){
    if(testModel.getNode(i).x()(0) > -.1)
      fx += f(i*3);
    if (testModel.getNode(i).x()(1) > 0.9)
      fy += f(i*3+1);
  }
  EXPECT_NEAR(fy, area*S11*(d+1.)/1., 1e-10);
  EXPECT_NEAR(fx, area*S22, 1e-10);
  */
}
