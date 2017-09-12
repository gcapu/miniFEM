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
  //Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dhdX; //used for precalculated data
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
  enum { 
      Dim = _Dim,
      NumNodes = _Dim == 3 ? 20 : 8, //quadratic elements only
      };
  typedef Node<Scalar, Dim> NodeType;
  typedef IntegrationPoint<Scalar, Dim> IPType;
  typedef std::array<int, NumNodes> ConnType; 
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
protected:
  ConnType _conn;
  const std::vector<NodeType>& _nodes; //global node storage. maybe I should change the name. W!
public:
  Element(const std::vector<NodeType>& nodes): _nodes(nodes){}
  virtual ~Element() {}
  //storage
  std::vector<IPType> ips; //integration points
  //access to dof values and nodes
  int Conn(int nodeNumber) const {return _conn[nodeNumber];}
  int Dof(int dofNumber) const {return Conn(dofNumber/Dim)*Dim + dofNumber % Dim;}
  const NodeType& Node(int nodeNumber) const {return _nodes.at(Conn(nodeNumber));}
  //set the connectivity from a container. returns true if succeeded.
  template <typename T> bool setConn(const T& conn);
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
  template <typename Derived>
  inline Eigen::Matrix<Scalar, Dim, 1> _u(const Eigen::MatrixBase<Derived>& naturalCoords) const;
  //h and _dhde return the shape functions and its derivatives wrt natural coords.
  template <typename Derived>
  inline Eigen::Matrix<Scalar, NumNodes, 1> _h(const Eigen::MatrixBase<Derived>& naturalCoords) const;
  template <typename Derived>
  inline Eigen::Matrix<Scalar, NumNodes, Dim> _dhde(const Eigen::MatrixBase<Derived>& naturalCoords) const;
public:
  C3D20R(const std::vector<NodeType>& nodes, const MatType& mat);
  virtual VectorType F();
  virtual VectorType LM();
  virtual MatrixType K();
  virtual MatrixType M();
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
  typedef Element<Scalar, Dim> ElementType;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::SparseMatrix<Scalar> MatrixType;
protected:
  std::vector<NodeType> _nodes;
  std::vector<std::unique_ptr<ElementType>> _elementPtrs; 
public:
  FEM(){};
  //Node and element access
  int NumNodes() const {return _nodes.size();}
  int NumElements() const {return _elementPtrs.size();}
  const NodeType& Node(int nodeNumber) const {return _nodes.at(nodeNumber);}
  const ElementType& Element(int elNumber) const {return *_elementPtrs.at(elNumber);} //W! might change to IP access
  //Update node displacements directly or using a global vector
  void setuNode(int i, const typename NodeType::VectorType& u) {_nodes.at(i).u = u;}
  void setu(const VectorType& u);
  //Obtain forces and
  VectorType F();
  VectorType LM();
  MatrixType K();
  MatrixType M();
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
bool Element<_Scalar, _Dim>::setConn(const T& conn){
  if(conn.size() < _conn.size()) return false;
  for(int i = 0; i<_conn.size(); i++) 
    _conn.at(i) = conn.at(i);
  return true;
  }

// ---- C3D20R ---- //

template <typename MatType>
C3D20R<MatType>::C3D20R(const std::vector<typename C3D20R<MatType>::NodeType>& nodes, const MatType& mat):
    Base(nodes), _mat(mat)
  {
  EIGEN_STATIC_ASSERT(MatType::Dim == 3, YOU_MADE_A_PROGRAMMING_MISTAKE);
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
  Eigen::Matrix<Scalar, Dim, NumNodes> allU;
  for(int i = 0; i<NumNodes; i++)
    allU.col(i) = this->node(i).u();
  return allU * shapeFunctions;  //I need to check the best return type.
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
  
template <typename MatType>
typename C3D20R<MatType>::VectorType C3D20R<MatType>::F() {
  //storage for the force
  VectorType fi = VectorType::Zero(NumDofs);
  Eigen::Map<Eigen::Matrix<double,NumNodes,Dim,Eigen::RowMajor>> Fi(fi.data());
  //obtaining uIi. Note that this is the transpose of belytschkos notation
  Eigen::Matrix<Scalar, Dim, NumNodes> u;
  for(int i = 0; i<NumNodes; i++)
    u.col(i) = this->Node(i).u;
  Eigen::Matrix3d H; //displacement gradient (not the deformation gradient)
  Eigen::Matrix<Scalar, Dim, NumNodes> allX;
  for(int i = 0; i<NumNodes; i++)
    allX.col(i) = this->Node(i).X;
  for (auto& ip: this->ips)
    {
    Eigen::Matrix<Scalar, NumNodes, Dim> dhdX;
    Eigen::Matrix<Scalar, NumNodes, Dim> dhde = _dhde(ip.natCoords);
    // F0e is the deformation gradient F^0_\psi between the material coords and natural coords
    Eigen::Matrix<Scalar, Dim, Dim> F0e = allX * dhde; 
    // detj is also between the material coords and natural coords
    ip.detj = F0e.determinant(); 
    //dhdX is called B^0_{Ij} by belytschko. (actually it's the transpose of this one)
    //while dhde are the same in all elements, that is not the case for dhdX.
    dhdX.noalias() = dhde * F0e.inverse(); //up to (including) this line everything could be precalculated
    H.noalias() = u*dhdX; //definition in Belytschko
    ip.strain.noalias() = .5*(H + H.transpose() + H*H.transpose()); //E
    ip.stress = _mat.Stress(ip.strain); //PK2
    Fi.noalias() += ip.detj * ip.weight * dhdX * ip.stress * (H.transpose() + Eigen::Matrix<Scalar, 3, 3>::Identity());
    }
  return fi;
  }


template <typename MatType>
typename C3D20R<MatType>::MatrixType C3D20R<MatType>::K() {
  //This function may or may not work. It's almost a placeholder. W!!

  //storage for the stiffness matrix
  MatrixType k = MatrixType::Zero(NumDofs, NumDofs);
  Eigen::Matrix<double, 6, 24> B; B.setZero();
  // W! repeated code with F() here:->
  Eigen::Matrix<Scalar, Dim, NumNodes> allX;
  for(int i = 0; i<NumNodes; i++)
    allX.col(i) = this->Node(i).X;
  for(int i = 0; i< this->ips.size(); i++) 
    {  
    typename Base::IPType& ip = this->ips.at(i);
    Eigen::Matrix<Scalar, NumNodes, Dim> dhdX;
    Eigen::Matrix<Scalar, NumNodes, Dim> dhde = _dhde(ip.natCoords);
    // F0e is the deformation gradient F^0_\psi between the material coords and natural coords
    Eigen::Matrix<Scalar, Dim, Dim> F0e = allX * dhde; 
    // detj is also between the material coords and natural coords
    ip.detj = F0e.determinant(); 
    //dhdX is called B^0_{Ij} by belytschko. (actually it's the transpose of this one)
    //while dhde are the same in all elements, that is not the case for dhdX.
    dhdX.noalias() = dhde * F0e.inverse(); //up to (including) this line everything could be precalculated
    // <--:repeated code up to here
        
    // The matrix B is just a container with the derivatives of h in X,Y,Z.
    //it is not necessary, but it makes easy the calculations.
    for(int i = 0;i < NumNodes; ++i) 
      {
      B(0, 3*i)     = dhdX(i,0);   
      B(1, 3*i + 1) = dhdX(i,1);
      B(2, 3*i + 2) = dhdX(i,2);
      B(3, 3*i + 1) = dhdX(i,2);
      B(3, 3*i + 2) = dhdX(i,1);
      B(4, 3*i)     = dhdX(i,2);
      B(4, 3*i + 2) = dhdX(i,0);
      B(5, 3*i)     = dhdX(i,1);
      B(5, 3*i + 1) = dhdX(i,0);
      }
    // Using the next function is actually slow. 
    Eigen::Matrix<double,6,6> Cm = _mat.Stiffness(); 
    Cm(3,3) /=2; //Given this definition of B we need the C that relates sigma and gamma    
    Cm(4,4) /=2; //Given this definition of B we need the C that relates sigma and gamma    
    Cm(5,5) /=2; //Given this definition of B we need the C that relates sigma and gamma    
    k+=B.transpose() * Cm * B * ip.detj * ip.weight; 
    }
  return k;
  }
  
template <typename MatType>
typename C3D20R<MatType>::MatrixType C3D20R<MatType>::M() {
  //This function may or may not work. It's almost a placeholder. W!!

  //storage for the stiffness matrix
  MatrixType m = MatrixType::Zero(NumDofs, NumDofs);
  
  for(int i = 0; i<this->ips.size(); i++)
    {
    typename Base::IPType& ip = this->ips.at(i);
    Eigen::Matrix<Scalar, NumNodes, 1> h = _h(ip.natCoords);
    // The matrix H is just a container with the values of h at the ips.
    Eigen::Matrix<double,Dim,NumDofs> H; H.setZero();
    // When filling the matrix, also the zero values must be initialized.
    for(int j = 0; j < NumNodes; ++j)
      {
      H(0, 3*j    ) = h(j);
      H(1, 3*j + 1) = h(j);
      H(2, 3*j + 2) = h(j);
      }
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

//sets u at the nodes given a global vector u with all the displacements
template <typename _Scalar, int _Dim>
void FEM<_Scalar, _Dim>::setu(const VectorType& u){
  for(int i = 0; i< NumNodes(); i++)
    setuNode(i, u.segment<Dim>(i*Dim));
}

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::VectorType FEM<_Scalar, _Dim>::F () {
  int numDofs = Dim * _nodes.size();
  VectorType f = VectorType::Zero(numDofs);
  for(auto& elementPtr: _elementPtrs) 
    {
    VectorType fel = elementPtr->F();
    for(int i = 0; i<fel.size(); i++)
      //if(dofId != dummy_dof)
      f(elementPtr->Dof(i)) += fel(i);
    }
  return f;
  }

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::VectorType FEM<_Scalar, _Dim>::LM () {
  int numDofs = Dim * _nodes.size();
  VectorType lm = VectorType::Zero(numDofs);
  for(auto& elementPtr: _elementPtrs)
    {
    VectorType lmel = elementPtr->UpdateLM();
    for(int i = 0; i<lmel.size(); i++)
      //if(dofId != dummy_dof)
      lm(elementPtr->Dof(i)) += lm(i);
    }
  return lm;
  }

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::MatrixType FEM<_Scalar, _Dim>::K () {
  MatrixType k;
  typedef Eigen::Triplet<Scalar> Trip;
  std::vector<Trip> tripletList;
  //size_t estimation_of_entries = elementPtrs.size() * elementPtrs.at(0)->ndofs() * elementPtrs.at(0)->ndofs();
  //tripletList.reserve(estimation_of_entries);
  for(auto& elementPtr: _elementPtrs)
    {
    typename ElementType::MatrixType elk = elementPtr->K();
    for(int i = 0; i<elk.rows(); i++) 
      //if(id1 != dummy_dof)
      for(int j = 0; j<elk.cols(); j++) 
        //if (id2 != dummy_dof)
        tripletList.push_back(Trip(elementPtr->Dof(i), elementPtr->Dof(j), elk(i, j)));
    }
  k.setFromTriplets(tripletList.begin(), tripletList.end());
  return k;
  }

template <typename _Scalar, int _Dim>
typename FEM<_Scalar, _Dim>::MatrixType FEM<_Scalar, _Dim>::M () {
  MatrixType m;
  typedef Eigen::Triplet<Scalar> Trip;
  std::vector<Trip> tripletList;
  //m.setZero(); <- should I?
  for(auto& elementPtr: _elementPtrs) 
    {
    typename ElementType::MatrixType elm = elementPtr->M();
    for(int i = 0; i<elm.rows(); i++)
      //if(id1 != dummy_dof)
      for(int j = 0; j<elm.cols(); j++)
        //if(id2 != dummy_dof)
        tripletList.push_back(Trip(elementPtr->Dof(i), elementPtr->Dof(j), elm(i, j)));
    }
  m.setFromTriplets(tripletList.begin(), tripletList.end()); 
  return m;
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
    _elementPtrs.emplace_back(new C3D20R<MatType>(_nodes, mat));
    //converting to index base 0
    for (int j = 0; j<ElDim; j++)
      tempEls[i][j] -= 1;
    _elementPtrs.at(i)->setConn(tempEls[i]);
    }
  return true;
  }


} //namespace TC
