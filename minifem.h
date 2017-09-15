/*---------------------------------------------------------------------------/
MIT License

Copyright (c) 2017 German Capuano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
/----------------------------------------------------------------------------*/
#pragma once

#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace mini
{
//----------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------

template <typename _Scalar, int _Dim>
struct Node{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  enum {Dim = _Dim};
  typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;
  Node(const VectorType& newX = VectorType::Zero(), 
       const VectorType& newu = VectorType::Zero()):
    X(newX), u(newu) {} 
  //Storage
  VectorType X;
  VectorType u;
  //These two let you use the updated position.
  VectorType getx() const {return X + u;}
  void setx(const VectorType& x) {u = x - X;}
  };

//----------------------------------------------------------------------------
// Integration points
//----------------------------------------------------------------------------

//The IntegrationPoint is in charge of providing its natural coordinates 
//    and weight, and to store info at their actual location. 
template <typename _Scalar, int _Dim>
struct IntegrationPoint{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  enum {Dim = _Dim};
  typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Dim, Dim> MatrixType;
  IntegrationPoint(const VectorType& NaturalCoords, double Weight):
      natCoords(NaturalCoords), weight(Weight) {} 
  //Storage variables
  const VectorType natCoords; //it's a bit wasteful to store this, but it's easier.
  const Scalar weight;
  VectorType X;
  MatrixType strain;
  MatrixType stress;
  Scalar detj;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dhdX; //used for precalculated data
};

//Integration rule types (most aren't implemented yet)
enum{prism, tet, quad, tri};

//Integration Rules provide sets of values to construct integration points.
//    There are better ways to implement it but this will do.
template <typename _Scalar, int type, int rule = 2>
struct IntegrationRule;

template <typename _Scalar>
struct IntegrationRule<_Scalar, prism, 2>{
  typedef _Scalar Scalar;
  enum {
    Dim = 3,
    NumPoints = 8,
    };
  typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;
protected:
  Eigen::Matrix<Scalar, Dim, NumPoints> points;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IntegrationRule(){
    Scalar d = 1/sqrt(3.);
    points << -d, d, d,-d,-d, d, d,-d,
              -d,-d, d, d,-d,-d, d, d,
              -d,-d,-d,-d, d, d, d, d;
    }
  VectorType point(int index) const { return points.col(index); }
  Scalar weight(int index = 0) const { return 1.; }
  };

//----------------------------------------------------------------------------
// Element
//----------------------------------------------------------------------------

//Base for all elements
template <typename _Scalar, int _Dim>
class Element{
public:
  typedef _Scalar Scalar;
  enum {Dim = _Dim};
  typedef Node<Scalar, Dim> NodeType;
  typedef IntegrationPoint<Scalar, Dim> IPType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
protected:
  const std::vector<NodeType>& _globalNodes; //reference to global node storage.
  std::vector<int> _connectivity;
public:
  template<typename T>
  Element(const std::vector<NodeType>& nodes, const T& conn);
  virtual ~Element() {}
  //storage
  std::vector<IPType> ips; //integration points
  //access to dof values and nodes
  int NumNodes() const {return _connectivity.size();}
  int NumDofs() const {return _connectivity.size() * Dim;}
  const NodeType& Node(int nodeNumber) const {return _globalNodes.at(Conn(nodeNumber));}
  int Conn(int nodeNumber) const {return _connectivity[nodeNumber];}
  const std::vector<int>& Conn() const {return _connectivity;}
  //main functions
  virtual VectorType F() = 0;
  virtual VectorType LM() = 0;
  virtual MatrixType K() = 0;
  virtual MatrixType M() = 0;
  };

//Quadratic 3D element with 20 nodes and reduced integration.
//For additional computaitonal savings, some terms relating the natural
//  and material coordinates at the integration points could be precalculated.
template <typename MatType>
class C3D20R: public Element<typename MatType::Scalar, 3>{
public:
  enum {
    Dim = 3,
    NumNodes = 20,
    NumDofs = Dim*NumNodes,
    IntegrationOrder = 2,
    NumIPs = 8
  };
  typedef Element<typename MatType::Scalar, Dim> Base;
  typedef typename Base::Scalar Scalar;
  typedef typename Base::NodeType NodeType;
  typedef typename Base::VectorType VectorType;
  typedef typename Base::MatrixType MatrixType;
  typedef IntegrationRule<Scalar, prism, 2> IntegrationRule;
protected:
  MatType _mat; //the material
  //h and _dhde return the shape functions and its derivatives wrt natural coords.
  template <typename Derived>
  inline Eigen::Matrix<Scalar, NumNodes, 1> _h(const Eigen::MatrixBase<Derived>& naturalCoords) const;
  template <typename Derived>
  inline Eigen::Matrix<Scalar, NumNodes, Dim> _dhde(const Eigen::MatrixBase<Derived>& naturalCoords) const;
  template <typename Derived>
  inline Eigen::Matrix<Scalar, Dim, 1> _u(const Eigen::MatrixBase<Derived>& naturalCoords) const;
  template <typename Derived>
  inline Eigen::Matrix<Scalar, NumNodes, Dim> _dhdX(const Eigen::MatrixBase<Derived>& naturalCoords) const;
public:
  template <typename T>
  C3D20R(const std::vector<NodeType>& nodes, const T& conn, const MatType& mat);
  virtual VectorType F();
  virtual VectorType LM();
  virtual MatrixType K();
  virtual MatrixType M();
  };

//----------------------------------------------------------------------------
// Constraints and Loads
//----------------------------------------------------------------------------

//Constraints are restrict the node dofs indicated in locked. It is applied to all 
//  nodes with 
template <int _Dim>
struct Constraint{
  enum {Dim = _Dim,};
  std::vector<int> nodeIds; //reference to global node storage.
  std::array<bool, Dim> locked = {true, true, true}; //by default it is an encastre
};

template <typename _Scalar, int _Dim>
struct Load{
  typedef _Scalar Scalar;
  enum {Dim = _Dim,};
  std::vector<int> nodeIds; //reference to global node storage.
  //force is the force applied to all nodes in the list. Since it is the same 
  //  exact value for all nodes it doesn't lead to uniform pressure (unfortunately).
  Eigen::Matrix<Scalar, Dim, 1> force; 
};

//----------------------------------------------------------------------------
// Finite element model
//----------------------------------------------------------------------------
template <typename _Scalar = double, int _Dim = 3>
class FEM{
public:
  typedef _Scalar Scalar;
  enum {
    Dim = _Dim,
    ElDim = _Dim == 3 ? 20 : 8, //quadratic elements only
    };
  typedef Node<Scalar, Dim> NodeType;
  typedef Constraint<Dim> ConstraintType;
  typedef Load<Scalar, Dim> LoadType;
  typedef Element<Scalar, Dim> ElementType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::SparseMatrix<Scalar> MatrixType;
protected:
  std::vector<NodeType> _nodes;
  std::vector<ConstraintType> _constraints;
  std::vector<LoadType> _loads;
  std::vector<std::unique_ptr<ElementType>> _elementPtrs; 
  //_constrainedDofIds maps the unconstrained degrees of freedom to a reduced
  //    system. Constrained dofs are indicated with negative values.
  std::vector<int> _reducedDofIds; 
  //unconstrained size is the number of free degrees of freedom.
  int _numFreeDofs; 
  void UpdateReduced();
  //GetElementReducedDofIds provides the reduced dof ids corresponding to its dofs.
  std::vector<int> GetElementReducedDofIds(int elementId) const;
public:
  FEM(){};
  //Access info
  int NumDofs() const {return _nodes.size()*Dim;}
  int NumFreeDofs() const {return _numFreeDofs;}
  int NumNodes() const {return _nodes.size();}
  int NumElements() const {return _elementPtrs.size();}
  const ElementType& Element(int elNumber) const {return *_elementPtrs.at(elNumber);} //W! might change to IP access
  const NodeType& Node(int nodeNumber) const {return _nodes.at(nodeNumber);}
  //Update node displacements directly or using a global vector
  void setuNode(int nodeNumber, const typename NodeType::VectorType& u) {_nodes.at(nodeNumber).u = u;}
  void setu(const VectorType& u);
  //Obtain forces and
  VectorType F();
  VectorType LM();
  MatrixType K();
  MatrixType M();
  //feature creation
  bool CreateConstraint(const ConstraintType& constraint);
  bool CreateLoad(const LoadType& load);
  template <typename MatType>
  bool ReadAbaqusInp(const std::string& filename, const MatType& mat);
  };

//----------------------------------------------------------------------------
// Sample Material (more at github.com/copono/TheConstitution)
//----------------------------------------------------------------------------

template <typename _Scalar, int _Dim>
class IsotropicLinear {
public:
  typedef _Scalar Scalar;
  enum {
    Dim = _Dim,
    StiffDim = _Dim==3?6:3
  };
  typedef Eigen::Matrix<Scalar, Dim, Dim> MatrixType;
  typedef Eigen::Matrix<Scalar, StiffDim, StiffDim> StiffType;
protected:
  Scalar _E;
  Scalar _nu;
  Scalar _density;
public:
  IsotropicLinear(Scalar E, Scalar nu, Scalar density = 1.): _E(E), _nu(nu), _density(density)
  {};
  template<typename Derived>
  MatrixType Stress(const Eigen::MatrixBase<Derived>& strain) const;
  StiffType Stiffness() const;
  Scalar E() const { return _E; }
  Scalar nu() const { return _nu; }
  Scalar G() const {return .5*E() / (1 + nu());}
  Scalar density() const {return _density;}
};
  
template <typename _Scalar, int _Dim>
typename IsotropicLinear<_Scalar, _Dim>::StiffType
IsotropicLinear<_Scalar, _Dim>::Stiffness() const
  {
  StiffType K = StiffType::Zero();
  Scalar ee = E()/((1+nu())*(1-2*nu()));
  K.template topLeftCorner<Dim, Dim>() = MatrixType::Constant(nu()*ee) + (1-2*nu())*ee * MatrixType::Identity();
  K.diagonal().template tail<StiffDim - Dim>() =  Eigen::Matrix<Scalar, StiffDim - Dim, 1>::Constant(G());
  return K;
  }
  
template <typename _Scalar, int _Dim>
template <typename Derived>
typename IsotropicLinear<_Scalar, _Dim>::MatrixType
    IsotropicLinear<_Scalar, _Dim>::Stress(const Eigen::MatrixBase<Derived>& strain) const
  {
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, MatrixType);
  MatrixType S;
  Scalar ee = E()/((1.+nu())*(1.-2.*nu()));
  Eigen::Matrix<_Scalar, Dim, 1> diag = (MatrixType::Constant(nu()*ee) + (1-2*nu())*ee * MatrixType::Identity()) * strain.derived().diagonal();
  S = (MatrixType::Constant(G()) - G() * MatrixType::Identity()).cwiseProduct(strain.derived());
  S += diag.asDiagonal();
  return S;// ;
  }
  
//----------------------------------------------------------------------------
// Helpers
//----------------------------------------------------------------------------
//Converts a string to int, float or double.
template <typename T> T convert(const std::string& val);

template <> int convert<int>(const std::string& val) {
  long temp = strtol(val.c_str(), NULL, 10);
  assert(temp <= INT_MAX);
  return static_cast<int>(temp) ;
}
template <> float convert<float>(const std::string& val) {
  return strtof(val.c_str(), NULL);
}
template <> double convert<double>(const std::string& val) {
  return strtod(val.c_str(), NULL);
}
bool strStartsWith(const std::string& str, const std::string& start){
  if (start.size() <= str.size() && std::equal(start.begin(), start.end(), str.begin()))
    return true;
  else
    return false;
}

//Appends the tokens generated from line into the tokens vector.
//An optional number of values can be ignored.
//Returns true if the line ends with a separator, false otherwise.
template <typename T>
bool tokenize(std::vector<T>& tokens, const std::string& line, int ignore = 0, char separator = ','){
  std::stringstream lineStream(line);
  std::string cell;
  //ingnore the first few
  for(int i = 0; i<ignore; i++)
    std::getline(lineStream, cell, separator);
  //split the line into tokens
  while (std::getline(lineStream, cell, separator))
    if(cell != "\r")
      tokens.push_back(convert<T>(cell));
  return !lineStream && (cell.empty() || cell == "\r"); 
}

//----------------------------------------------------------------------------
// function definitions
//----------------------------------------------------------------------------

// ---- Element ---- //

template <typename _Scalar, int _Dim>
template <typename T>
Element<_Scalar, _Dim>::Element(const std::vector<NodeType>& nodes, const T& conn):
    _globalNodes(nodes)
  {
  if(_connectivity.size() != conn.size())
    _connectivity.resize(conn.size());
  for(int i = 0; i<_connectivity.size(); i++) 
    _connectivity.at(i) = conn.at(i);
  }
  
// ---- C3D20R ---- //

template <typename MatType>
template <typename T>
C3D20R<MatType>::C3D20R(const std::vector<NodeType>& nodes,
                        const T& conn, const MatType& mat):
    Base(nodes, conn), _mat(mat)
  {
  EIGEN_STATIC_ASSERT(MatType::Dim == 3, YOU_MADE_A_PROGRAMMING_MISTAKE);
  assert(conn.size() == NumNodes);
  IntegrationRule IR;
  for(int i = 0; i < IntegrationRule::NumPoints; i++)
    this->ips.emplace_back(IR.point(i), IR.weight(i));
  }

template <typename MatType>
template <typename Derived>
Eigen::Matrix<typename C3D20R<MatType>::Scalar, C3D20R<MatType>::NumNodes, 1>
    C3D20R<MatType>::_h(const Eigen::MatrixBase<Derived>& naturalCoords) const
  {
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived, Eigen::Vector3d); //we're just checking size so any scalar type is ok.
  Scalar g = naturalCoords(0);
  Scalar h = naturalCoords(1);
  Scalar r = naturalCoords(2);
  Eigen::Matrix<Scalar, NumNodes, 1> shapeFunctions;
  shapeFunctions << -1/8*(1-g)*(1-h)*(1-r)*(2+g+h+r),
                    -1/8*(1+g)*(1-h)*(1-r)*(2-g+h+r),
                    -1/8*(1+g)*(1+h)*(1-r)*(2-g-h+r),
                    -1/8*(1-g)*(1+h)*(1-r)*(2+g-h+r),
                    -1/8*(1-g)*(1-h)*(1+r)*(2+g+h-r),
                    -1/8*(1+g)*(1-h)*(1+r)*(2-g+h-r),
                    -1/8*(1+g)*(1+h)*(1+r)*(2-g-h-r),
                    -1/8*(1-g)*(1+h)*(1+r)*(2+g-h-r),
                    1/4*(1-g)*(1+g)*(1-h)*(1-r),
                    1/4*(1-h)*(1+h)*(1+g)*(1-r),
                    1/4*(1-g)*(1+g)*(1+h)*(1-r),
                    1/4*(1-h)*(1+h)*(1-g)*(1-r),
                    1/4*(1-g)*(1+g)*(1-h)*(1+r),
                    1/4*(1-h)*(1+h)*(1+g)*(1+r),
                    1/4*(1-g)*(1+g)*(1+h)*(1+r),
                    1/4*(1-h)*(1+h)*(1-g)*(1+r),
                    1/4*(1-r)*(1+r)*(1-g)*(1-h),
                    1/4*(1-r)*(1+r)*(1+g)*(1-h),
                    1/4*(1-r)*(1+r)*(1+g)*(1+h),
                    1/4*(1-r)*(1+r)*(1-g)*(1+h);
  return shapeFunctions; 
}

template <typename MatType>
template <typename Derived>
Eigen::Matrix<typename C3D20R<MatType>::Scalar, C3D20R<MatType>::Dim, 1>
    C3D20R<MatType>::_u(const Eigen::MatrixBase<Derived>& naturalCoords) const
  {
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived, Eigen::Vector3d); //Any scalar type should be ok since only size matters.
  Eigen::Matrix<Scalar, NumNodes, 1> shapeFunctions = _h(naturalCoords);
  Eigen::Matrix<Scalar, Dim, NumNodes> allU;
  for(int i = 0; i<NumNodes; i++)
    allU.col(i) = this->node(i).u();
  return allU * shapeFunctions;
}

template <typename MatType>
template <typename Derived>
Eigen::Matrix<typename C3D20R<MatType>::Scalar, C3D20R<MatType>::NumNodes,C3D20R<MatType>::Dim>
    C3D20R<MatType>::_dhde(const Eigen::MatrixBase<Derived>& naturalCoords) const
  {
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived, Eigen::Vector3d); //Any scalar type should work.
  Scalar g = naturalCoords(0);
  Scalar h = naturalCoords(1);
  Scalar r = naturalCoords(2);
  Eigen::Matrix<Scalar, NumNodes, Dim> dhde;
  dhde <<
   (1-h)*(1-r)*(1+2*g+h+r)/8., (1-g)*(1-r)*(1+g+2*h+r)/8., (1-g)*(1-h)*(1+g+h+2*r)/8.,
  -(1-h)*(1-r)*(1-2*g+h+r)/8., (1+g)*(1-r)*(1-g+2*h+r)/8., (1+g)*(1-h)*(1-g+h+2*r)/8.,
  -(1+h)*(1-r)*(1-2*g-h+r)/8.,-(1+g)*(1-r)*(1-g-2*h+r)/8., (1+g)*(1+h)*(1-g-h+2*r)/8.,
   (1+h)*(1-r)*(1+2*g-h+r)/8.,-(1-g)*(1-r)*(1+g-2*h+r)/8., (1-g)*(1+h)*(1+g-h+2*r)/8.,
   (1-h)*(1+r)*(1+2*g+h-r)/8., (1-g)*(1+r)*(1+g+2*h-r)/8.,-(1-g)*(1-h)*(1+g+h-2*r)/8.,
  -(1-h)*(1+r)*(1-2*g+h-r)/8., (1+g)*(1+r)*(1-g+2*h-r)/8.,-(1+g)*(1-h)*(1-g+h-2*r)/8.,
  -(1+h)*(1+r)*(1-2*g-h-r)/8.,-(1+g)*(1+r)*(1-g-2*h-r)/8.,-(1+g)*(1+h)*(1-g-h-2*r)/8.,
   (1+h)*(1+r)*(1+2*g-h-r)/8.,-(1-g)*(1+r)*(1+g-2*h-r)/8.,-(1-g)*(1+h)*(1+g-h-2*r)/8.,
   ( -g)*(1-h)*(1-r)/2.,      -(1-g)*(1+g)*(1-r)/4.,      -(1-g)*(1+g)*(1-h)/4.,
   (1-h)*(1+h)*(1-r)/4.,       ( -h)*(1+g)*(1-r)/2.,      -(1-h)*(1+h)*(1+g)/4.,
   ( -g)*(1+h)*(1-r)/2.,       (1-g)*(1+g)*(1-r)/4.,      -(1-g)*(1+g)*(1+h)/4.,
  -(1-h)*(1+h)*(1-r)/4.,       ( -h)*(1-g)*(1-r)/2.,      -(1-h)*(1+h)*(1-g)/4.,
   ( -g)*(1-h)*(1+r)/2.,      -(1-g)*(1+g)*(1+r)/4.,       (1-g)*(1+g)*(1-h)/4.,
   (1-h)*(1+h)*(1+r)/4.,       ( -h)*(1+g)*(1+r)/2.,       (1-h)*(1+h)*(1+g)/4.,
   ( -g)*(1+h)*(1+r)/2.,       (1-g)*(1+g)*(1+r)/4.,       (1-g)*(1+g)*(1+h)/4.,
  -(1-h)*(1+h)*(1+r)/4.,       ( -h)*(1-g)*(1+r)/2.,       (1-h)*(1+h)*(1-g)/4.,
  -(1-r)*(1+r)*(1-h)/4.,      -(1-r)*(1+r)*(1-g)/4.,       ( -r)*(1-g)*(1-h)/2.,
   (1-r)*(1+r)*(1-h)/4.,      -(1-r)*(1+r)*(1+g)/4.,       ( -r)*(1+g)*(1-h)/2.,
   (1-r)*(1+r)*(1+h)/4.,       (1-r)*(1+r)*(1+g)/4.,       ( -r)*(1+g)*(1+h)/2.,
  -(1-r)*(1+r)*(1+h)/4.,       (1-r)*(1+r)*(1-g)/4.,       ( -r)*(1-g)*(1+h)/2.;
  return dhde;
}

//W!! this function will change. Right now detj0 is discarded. We want it.
template <typename MatType>
template <typename Derived>
Eigen::Matrix<typename C3D20R<MatType>::Scalar, C3D20R<MatType>::NumNodes,C3D20R<MatType>::Dim>
    C3D20R<MatType>::_dhdX(const Eigen::MatrixBase<Derived>& naturalCoords) const
  {
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived, Eigen::Vector3d); //Any scalar type should work.
  Eigen::Matrix<Scalar, Dim, NumNodes> allX;
  for(int i = 0; i<NumNodes; i++)
    allX.col(i) = this->Node(i).X;
  Eigen::Matrix<Scalar, NumNodes, Dim> dhdX;
  Eigen::Matrix<Scalar, NumNodes, Dim> dhde = _dhde(naturalCoords);
  // F0e is the deformation gradient F^0_\psi between the material coords and natural coords
  Eigen::Matrix<Scalar, Dim, Dim> F0e = allX * dhde; 
  // detj is also between the material coords and natural coords
  Scalar detj = F0e.determinant(); 
  //dhdX is called B^0_{Ij} by belytschko. (actually it's the transpose of this one)
  //while dhde are the same in all elements, that is not the case for dhdX.
  dhdX.noalias() = dhde * F0e.inverse(); //up to (including) this line everything could be precalculated
  return dhdX;
  }
  
template <typename MatType>
typename C3D20R<MatType>::VectorType C3D20R<MatType>::F() {
  //storage for the force
  VectorType fi = VectorType::Zero(NumDofs);
  Eigen::Map<Eigen::Matrix<Scalar,NumNodes,Dim,Eigen::RowMajor>> Fi(fi.data());
  //obtaining uIi. Note that this is the transpose of belytschkos notation
  Eigen::Matrix<Scalar, Dim, NumNodes> allu;
  for(int i = 0; i<NumNodes; i++)
    allu.col(i) = this->Node(i).u;
  Eigen::Matrix<Scalar, Dim, NumNodes> allX;
  for(int i = 0; i<NumNodes; i++)
    allX.col(i) = this->Node(i).X;
  for (auto& ip: this->ips)
    {
    Eigen::Matrix<Scalar, NumNodes, Dim> dhde = _dhde(ip.natCoords);
    // F0e is the deformation gradient F^0_\psi between the material coords and natural coords
    Eigen::Matrix<Scalar, Dim, Dim> F0e = allX * dhde; 
    // detj is also between the material coords and natural coords
    ip.detj = F0e.determinant(); 
    //dhdX is called B^0_{Ij} by belytschko. (actually it's the transpose of this one)
    //while dhde are the same in all elements, that is not the case for dhdX.
    Eigen::Matrix<Scalar, NumNodes, Dim> dhdX = dhde * F0e.inverse(); //up to (including) this line everything could be precalculated
    Eigen::Matrix<Scalar, Dim, Dim> H = allu * dhdX; ////displacement gradient (not the deformation gradient)
    ip.strain.noalias() = .5*(H + H.transpose() + H*H.transpose()); //E
    ip.stress = _mat.Stress(ip.strain); //PK2
    Fi.noalias() += ip.detj * ip.weight * dhdX * ip.stress * (H.transpose() + Eigen::Matrix<Scalar, 3, 3>::Identity());
    }
  return fi;
  }


template <typename MatType>
typename C3D20R<MatType>::MatrixType C3D20R<MatType>::K() {
  //storage for the stiffness matrix
  MatrixType k = MatrixType::Zero(NumDofs, NumDofs);
  Eigen::Matrix<Scalar, 6, NumDofs> B; B.setZero(); //W! hardcoded number
  // W! repeated code with F() here:->
  Eigen::Matrix<Scalar, Dim, NumNodes> allX;
  Eigen::Matrix<Scalar, Dim, NumNodes> allu;
  for(int i = 0; i<NumNodes; i++)
    allX.col(i) = this->Node(i).X;
  for(int i = 0; i<NumNodes; i++)
    allu.col(i) = this->Node(i).u;
  for(int i = 0; i< this->ips.size(); i++) 
    {  
    typename Base::IPType& ip = this->ips.at(i);
    Eigen::Matrix<Scalar, NumNodes, Dim> dhde = _dhde(ip.natCoords);
    // F0e is the deformation gradient F^0_\psi between the material coords and natural coords
    Eigen::Matrix<Scalar, Dim, Dim> F0e = allX * dhde; 
    // detj is also between the material coords and natural coords
    ip.detj = F0e.determinant(); 
    //dhdX is called B^0_{Ij} by belytschko. (actually it's the transpose of this one)
    //while dhde are the same in all elements, that is not the case for dhdX.
    Eigen::Matrix<Scalar, NumNodes, Dim> dhdX = dhde * F0e.inverse(); //up to (including) this line everything could be precalculated
    Eigen::Matrix<Scalar, Dim, Dim> H = allu * dhdX; ////displacement gradient (not the deformation gradient)
    Eigen::Matrix<Scalar,Dim, Dim> F = H + Eigen::Matrix<Scalar, Dim, Dim>::Identity();
    ip.strain.noalias() = .5*(H + H.transpose() + H*H.transpose()); //E
    ip.stress = _mat.Stress(ip.strain); //PK2
     // <--:repeated code up to here
    // Maybe I should just do what belytschko recommends in page 214 and write the individual terms.
    for(int j = 0;j < NumNodes; ++j) 
      {
      B.template block<3,3>(0,j*Dim) = dhdX.row(i).asDiagonal() * F;
      B.template block<1,3>(3,j*Dim) = dhdX(j,1) * F.col(3).transpose() + dhdX(j,2) * F.col(2).transpose();
      B.template block<1,3>(4,j*Dim) = dhdX(j,0) * F.col(3).transpose() + dhdX(j,2) * F.col(1).transpose();
      B.template block<1,3>(5,j*Dim) = dhdX(j,0) * F.col(2).transpose() + dhdX(j,1) * F.col(1).transpose();
      }
    // Using the next function is actually slow. 
    Eigen::Matrix<double,6,6> Cm = _mat.Stiffness();  
    /*/Todo: veify if I need to do this W!!!
    Cm(3,3) /=2; //Given this definition of B we need the C that relates sigma and gamma    
    Cm(4,4) /=2; //Given this definition of B we need the C that relates sigma and gamma    
    Cm(5,5) /=2; //Given this definition of B we need the C that relates sigma and gamma    */
    k+=B.transpose() * Cm * B * ip.detj * ip.weight;
    Eigen::Matrix<Scalar, NumNodes, NumNodes> Kgeo = dhdX * ip.stress * dhdX.transpose() *ip.detj * ip.weight;
    for(int j = 0; j<NumNodes; j++)
      for(int l = 0; l<NumNodes; l++) //W! choice for indices is less than ideal given that k is a matrix
        k.template block<Dim, Dim>(Dim*j, Dim*l) += Kgeo(j,l) * Eigen::Matrix<Scalar, Dim, Dim>::Identity(); 
    }
  return k;
  }
  
template <typename MatType>
typename C3D20R<MatType>::MatrixType C3D20R<MatType>::M() {
  //storage for the stiffness matrix
  MatrixType m = MatrixType::Zero(NumDofs, NumDofs);
  for(int i = 0; i<this->ips.size(); i++)
    {
    typename Base::IPType& ip = this->ips.at(i);
    Eigen::Matrix<Scalar, NumNodes, 1> h = _h(ip.natCoords);
    // The matrix H is just a container with the values of h at the ips.
    Eigen::Matrix<double,Dim,NumDofs> H;
    // When filling the matrix, also the zero values must be initialized.
    for(int j = 0; j < NumNodes; ++j)
      H.template block<Dim,Dim>(0, j*Dim) = h(j) * Eigen::Matrix<Scalar, Dim, Dim>::Identity();
    // The next function might not be very optimized.
    m+= H.transpose() * H * _mat.density() * ip.detj * ip.weight; 
    }
  return m;
  }

template <typename MatType>
typename C3D20R<MatType>::VectorType C3D20R<MatType>::LM() {
  //storage for the force
  MatrixType m = M();
  VectorType lm = VectorType::Zero(NumDofs);
  //calculations
  for(int i = 0; i<NumDofs; i++)
    for(int j = 0; j<NumDofs; j++)
      lm(i)+=m(i,j);
  return lm;
  }

// ------ FEM ------ //

//Private method! Updates the _constrainedDofIds. 
template <typename _Scalar, int _Dim>
void FEM<_Scalar, _Dim>::UpdateReduced(){
  //filling with the default values
  if(_reducedDofIds.size() != NumDofs())
    _reducedDofIds.resize(NumDofs());
  for(int i = 0; i<NumDofs(); i++)
    _reducedDofIds[i] = i;
  //leaving a tag equal to -(i+1) where i is the constraint number on constrained dofs
  for(int i = 0; i<_constraints.size(); i++)
    for(int j = 0; j<Dim; j++)
      if(_constraints.at(i).locked[j])
        for(auto nodeId: _constraints.at(i).nodeIds)
          _reducedDofIds[Dim*nodeId+j] = -(i+1);
  //filling in the other dofs
  int dofId = 0;
  for(int i = 0; i<NumDofs(); i++)
    if(_reducedDofIds[i] >= 0)
      _reducedDofIds[i] = dofId++;
  _numFreeDofs = dofId; 
}

//Private method! gives the reduced dofs for the element
template <typename _Scalar, int _Dim>
std::vector<int> FEM<_Scalar, _Dim>::GetElementReducedDofIds(int elementId) const {
  std::vector<int> result;
  for(auto nodeId: Element(elementId).Conn())
    for(int j = 0; j < Dim; j++)
      result.push_back(_reducedDofIds.at(nodeId*Dim+j));
  return result;
}

//sets u at the nodes given a global vector u with all the displacements
template <typename _Scalar, int _Dim>
void FEM<_Scalar, _Dim>::setu(const VectorType& u){
  for(int i = 0; i< NumNodes(); i++)
    setuNode(i, u.segment<Dim>(i*Dim));
}

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::VectorType FEM<_Scalar, _Dim>::F () {
  VectorType f = VectorType::Zero(NumFreeDofs());
  for(int i = 0; i < NumElements(); i++) 
    {
    typename ElementType::VectorType fel = _elementPtrs.at(i)->F();
    std::vector<int> dofs = GetElementReducedDofIds(i);
    assert(fel.size() == dofs.size());
    for(int j = 0; j<dofs.size(); j++)
      if(dofs[j] >= 0)
        f(dofs[j]) += fel(j);
    }
  for(const auto& load: _loads)
    for(const auto& nodeId: load.nodeIds)
      for(int i = 0; i < Dim; i++)
        {
        int dofId = _reducedDofIds[nodeId*Dim+i];
        if(dofId >= 0)
          f(dofId) += load.force[i];
        }
  return f;
  }

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::VectorType FEM<_Scalar, _Dim>::LM () {
  VectorType lm = VectorType::Zero(NumFreeDofs());
  for(int i = 0; i < NumElements(); i++) 
    {
    typename ElementType::VectorType lmel = _elementPtrs.at(i)->UpdateLM();
    std::vector<int> dofs = GetElementReducedDofIds(i);
    assert(lmel.size() == dofs.size());
    for(int j = 0; j<dofs.size(); j++)
      if(dofs[j] >= 0)
        lm(dofs[j]) += lm(j);
    }
  return lm;
  }

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::MatrixType FEM<_Scalar, _Dim>::K () {
  MatrixType k(NumFreeDofs(), NumFreeDofs());
  typedef Eigen::Triplet<Scalar> Trip;
  std::vector<Trip> tripletList;
  //size_t estimation_of_entries = elementPtrs.size() * elementPtrs.at(0)->ndofs() * elementPtrs.at(0)->ndofs();
  //tripletList.reserve(estimation_of_entries);
  for(int i = 0; i < NumElements(); i++) 
    {
    typename ElementType::MatrixType kel = _elementPtrs.at(i)->K();
    std::vector<int> dofs = GetElementReducedDofIds(i);
    assert(kel.size() == dofs.size());
    for(int j = 0; j<dofs.size(); j++)
      if(dofs[j] >= 0)
        for(int k = 0; k<dofs.size(); k++) 
          if(dofs[k] >= 0 && kel(j,k) > Eigen::NumTraits<Scalar>::dummy_precision())
            tripletList.push_back(Trip(dofs[j], dofs[k], kel(j, k)));
    }
  k.setFromTriplets(tripletList.begin(), tripletList.end());
  return k;
  }

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::MatrixType FEM<_Scalar, _Dim>::M () {
  MatrixType m(NumFreeDofs(), NumFreeDofs());
  typedef Eigen::Triplet<Scalar> Trip;
  std::vector<Trip> tripletList;
  for(int i = 0; i < NumElements(); i++) 
    {
    typename ElementType::MatrixType mel = _elementPtrs.at(i)->M();
    std::vector<int> dofs = GetElementReducedDofIds(i);
    assert(mel.size() == dofs.size());
    for(int j = 0; j<dofs.size(); j++)
      if(dofs[j] >= 0)
        for(int k = 0; k<dofs.size(); k++) 
          if(dofs[k] >= 0 && mel(j,k) > Eigen::NumTraits<Scalar>::dummy_precision())
            tripletList.push_back(Trip(dofs[j], dofs[k], mel(j, k)));
    }
  m.setFromTriplets(tripletList.begin(), tripletList.end()); 
  return m;
  }
template <typename _Scalar, int _Dim>
bool FEM<_Scalar, _Dim>::CreateConstraint(const ConstraintType& constraint){
  _constraints.push_back(constraint);
  UpdateReduced();
}
template <typename _Scalar, int _Dim>
bool FEM<_Scalar, _Dim>::CreateLoad(const LoadType& load){
  _loads.push_back(load);
}


//This function can read the mesh from an Abaqus input file. It's a very 
//  basic implementation so don't get too crazy about it. It blindly trusts
//  that the connectivity matches the value of FEM::NumNodes.
//Returns true if succeeded
template <typename _Scalar, int _Dim>
template <typename MatType>
bool FEM<_Scalar, _Dim>::ReadAbaqusInp(const std::string& filename, const MatType& mat){
  //Setup
  std::vector<std::vector<Scalar>> tempNodes;
  std::vector<std::vector<int>> tempEls;
  std::ifstream inf(filename.c_str());
  if(!inf)
    {
    std::cerr << "Warning: Can't read input file." << std::endl;
    return false;
    }
  std::string line;
  std::getline(inf, line);
  //Ignore lines until a node command
  while (!strStartsWith(line, "*Node"))
    std::getline(inf, line);
  //Read the first node and check if file is invalid
  std::getline(inf, line);
  if(strStartsWith(line, "*"))
    {
    std::cerr << "Warning: Input file contains a command when node was expected." << std::endl;
    return false;
    }
  //get nodes while line is not a command
  while(!strStartsWith(line, "*"))
    {
    tempNodes.push_back(std::vector<Scalar>()); 
    tokenize(tempNodes.back(), line,1);
    std::getline(inf, line);
    }
  //Make sure it's the right dimension
  for(int i = 0; i<tempNodes.size(); i++)
    if(tempNodes[i].size() != Dim)
      {
      std::cerr << "Warning: Size of nodes doesn't match model dimension." << std::endl;
      return false;
      }
  //Ignore lines until an Element command
  while (!strStartsWith(line, "*Element"))
    std::getline(inf, line);
  //Read the first Element and check if file is invalid
  std::getline(inf, line);
  if (strStartsWith(line, "*"))
    {
    std::cerr << "Warning: Input file contains a command when element was expected." << std::endl;
    return false;
    }
  //get Elements while line is not a command
  while (!strStartsWith(line, "*"))
    {
    tempEls.push_back(std::vector<int>());
    bool endsWithComma = tokenize(tempEls.back(), line, 1);
    std::getline(inf, line);
    //While the line ends with comma we consider it to continue next line
    while (endsWithComma) {
      endsWithComma = tokenize(tempEls.back(), line, 0);
      std::getline(inf, line);
      }
    }
  //Checking the dimensions are right and that connectivity values are
  //  consistent with the number of nodes. Note that Abaqus uses a base
  //  index equals to 1.
  for(int i = 0; i<tempEls.size(); i++)
    {
    if (tempEls[i].size() != ElDim)
      {
      std::cerr << "Warning: Input file contains wrong connectivity size." << std::endl <<
          "Given " << tempEls[i].size() << " values when " << ElDim << " were expected." << std::endl;
      return false;
      }
    for (int j = 0; j<tempEls[i].size(); j++)
      if(tempEls[i][j] > tempNodes.size())
        {
        std::cerr << "Warning: Input file contains wrong connectivity values." << std::endl;
        return false;
        }
    }
  //Build the matrices. Since this is done after reading all the data
  //  and performing all the desired checks it's more unlikely that 
  //  changes are performed on the model during failure.
  _nodes.resize(tempNodes.size());
  for (int i = 0; i<tempNodes.size(); i++)
    {
    typename NodeType::VectorType X;
    for(int j=0; j<Dim; j++)
      X(j) = tempNodes[i][j];
    _nodes.at(i).X = X;
    }
  for (int i = 0; i<tempEls.size(); i++)
    {
    //converting to index base 0
    for (int j = 0; j<ElDim; j++)
      tempEls[i][j] -= 1;
    _elementPtrs.emplace_back(new C3D20R<MatType>(_nodes, tempEls[i], mat));
    }
  UpdateReduced();
  return true;
  }


} //namespace TC
