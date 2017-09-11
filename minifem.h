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

namespace mini
{
//----------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------

template <typename _Scalar, int _Dim>
struct Node{
  typedef _Scalar Scalar;
  enum {Dim = _Dim};
  typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;
  VectorType X;
  VectorType u;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Node(const VectorType& newX = VectorType::Zero(), 
       const VectorType& newu = VectorType::Zero()):
    X(newX), u(newu) {} 
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
  IntegrationPoint(const VectorType& NaturalCoords, double _weight):
      NatCoords(NaturalCoords), weight(_weight) {} 
  //Storage variables
  const VectorType NatCoords; //it's a bit wasteful to store this, but it's easier.
  const Scalar weight;
  VectorType X;
  MatrixType strain;
  MatrixType stress;
  Scalar detj;
};

//Integration rules (most aren't implemented yet)
enum{prism, tet, quad, tri};

//Integration Rules provide sets of values to construct integration points.
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
  VectorType point(int id) const { return points.col(id); }
  Scalar weight() const { return 1.; }
  };

//----------------------------------------------------------------------------
// Element
//----------------------------------------------------------------------------

// Storage for vairables at a particular point. W!
template <typename _Scalar, int _Dim>
struct PointInfo{
  enum {Dim = _Dim};
  typedef _Scalar Scalar;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix<Scalar, Dim, 1> X;
  Eigen::Matrix<Scalar, Dim, 1> u;
  Eigen::Matrix<Scalar, Dim, Dim> stress;
  Eigen::Matrix<Scalar, Dim, Dim> strain;
  Scalar elasticEnergy;
  Scalar kineticEnergy;
};
  
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
  //virtual VectorType LM() const = 0;
  //virtual MatrixType K() const = 0;
  //virtual MatrixType M() const = 0;
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
protected:
  MatType _mat; //the material
  template <typename Derived>
  inline Eigen::Matrix<Scalar, Dim, 1> _u(const Eigen::MatrixBase<Derived>& naturalCoords) const;
  //_dhde returns the derivatives of the shape functions wrt natural coords.
  template <typename Derived>
  inline Eigen::Matrix<Scalar, NumNodes, Dim> _dhde(const Eigen::MatrixBase<Derived>& naturalCoords) const;
public:
  C3D20R(const std::vector<NodeType>& nodes, const MatType& mat):
      Base(nodes), _mat(mat)
    {
    EIGEN_STATIC_ASSERT(MatType::Dim == 3, YOU_MADE_A_PROGRAMMING_MISTAKE);
    }
  virtual VectorType F() const;
  //VectorType LM() const;
  //MatrixType K() const;
  //MatrixType M() const;
  };

// I might change this and dhde for some constepxr W!
template <typename MatType>
const GaussRule<typename C3D20R<MatType>::Scalar, prism, C3D20R<MatType>::IntegrationOrder>
  C3D20R<MatType>::_ips = GaussRule<typename C3D20R<MatType>::Scalar, prism, C3D20R<MatType>::IntegrationOrder>(); //integration points

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
protected:
  std::vector<NodeType> _nodes;
  std::vector<std::unique_ptr<ElementType>> _elementPtrs; 
public:
  FEM(){};
  //Node and element access
  int NumNodes() const {return _nodes.size();}
  int NumElements() const {return _elementPtrs.size();}
  const NodeType& getNode(int nodeNumber) const {return _nodes.at(nodeNumber);}
  const ElementType& getElement(int elNumber) const {return *_elementPtrs.at(elNumber);}
  //Update node displacements directly or using a global vector
  void setuNode(int i, const typename NodeType::VectorType& u) {_nodes.at(i).setu(u);}
  void setu(const VectorType& u);
  //Obtain forces and
  VectorType F();
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
public:
  IsotropicLinear(Scalar E, Scalar nu): _E(E), _nu(nu)
  {};
  template<typename Derived>
  MatrixType Stress(const Eigen::MatrixBase<Derived>& strain) const;
  StiffType Stiffness() const;
  Scalar E() const { return _E; }
  Scalar nu() const { return _nu; }
  Scalar G() const {return .5*E() / (1 + nu());}
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
typename C3D20R<MatType>::VectorType C3D20R<MatType>::F(std::vector<PointInfo>& storage) const {
  //preparing storage for output
  if(storage.size() < NumIPs) storage.resize(NumIPs);
  //storage for the force
  VectorType fi = VectorType::Zero(NumDofs);
  Eigen::Map<Eigen::Matrix<double,NumNodes,Dim,Eigen::RowMajor>> Fi(fi.data());
  //obtaining uIi. Note that this is the transpose of belytschkos notation
  Eigen::Matrix<Scalar, Dim, NumNodes> u;
  for(int i = 0; i<NumNodes; i++)
    u.col(i) = this->getNode(i).u();
  Eigen::Matrix3d H; //displacement gradient (not the deformation gradient)
  Eigen::Matrix<Scalar, Dim, NumNodes> allX;
  for(int i = 0; i<NumNodes; i++)
    allX.col(i) = this->getNode(i).X();
  for (int i = 0; i<NumIPs; i++)
    {
    Eigen::Matrix<Scalar, NumNodes, Dim> dhdX;
    Eigen::Matrix<Scalar, NumNodes, Dim> dhde = _dhde(_ips.point(i));
    // F0e is the deformation gradient F^0_\psi between the material coords and element coords
    Eigen::Matrix<Scalar, Dim, Dim> F0e = allX * dhde; 
    Scalar detj0 = F0e.determinant(); 
    //dhdX is called B^0_{Ij} by belytschko. (actually it's the transpose of this one)
    //while dhde are the same in all elements, that is not the case for dhdX.
    dhdX = dhde * F0e.inverse(); //up to (including) this line everything could be precalculated
    H.noalias() = u*dhdX; //definition in Belytschko
    storage.at(i).strain.noalias() = .5*(H + H.transpose() + H*H.transpose()); //E
    storage.at(i).stress = _mat.Stress(storage.at(i).strain); //PK2
    double k = detj0 * _ips.weight();
    Fi.noalias() += k*dhdX * storage.at(i).stress * (H.transpose() + Eigen::Matrix<Scalar, 3, 3>::Identity());
    }
  return fi;
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
  std::vector<typename Element::PointInfo> storage;
  for(auto& elementPtr: _elementPtrs) 
    {
    VectorType fel = elementPtr->F(storage);
    for(int i = 0; i<fel.size(); i++)
      {
      //if(dofId != dummy_dof)
      f(elementPtr->getDof(i)) += fel(i);
      }
    }
  return f;
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
    typename Node::VectorType X;
    for(int j=0; j<Dim; j++)
      X(j) = tempNodes[i][j];
    _nodes.at(i).setX(X);
    }
  for (int i = 0; i<tempEls.size(); i++)
    {
    _elementPtrs.emplace_back(new C3D20R<MatType>(_nodes, mat));
    //converting to index base 0
    for (int j = 0; j<ElDim; j++)
      tempEls[i][j] -= 1;
    _elementPtrs.at(i)->setConn(tempEls[i]); //converting to index base 0
    }
  return true;
  }


} //namespace TC
