#include "benchmark/benchmark.h"
#include "minifem.h"


static void BM_mini(benchmark::State &state) {
  mini::FEM<double,3> mymodel;
  mini::IsotropicLinear<double, 3> li(200e9, .3);
  if(!mymodel.ReadAbaqusInp("../benchmark/input/bar-4x1x1-2el.inp", li))
    std::cout << "Warning: failed reading input file." << std::endl;
  while (state.KeepRunning()) {
    mymodel.F();
    }
  }

// Register the function as a benchmark
BENCHMARK(BM_mini);

//~~~~~~~~~~~~~~~~
static void BM_isolinear(benchmark::State &state) {
mini::IsotropicLinear<double, 2> mat(200, 0.3);
while (state.KeepRunning()) {
  mat.Stiffness();
  Eigen::Matrix2d E;
  E << 0.1, 0.05, 0.05, 0.05;
  mat.Stress(E);
  }
}

  // Register the function as a benchmark
  BENCHMARK(BM_isolinear);

  //~~~~~~~~~~~~~~~~
static void BM_isolinear_eval(benchmark::State &state) {
mini::IsotropicLinear<double, 2> mat(200, 0.3);
while (state.KeepRunning()) {
  mat.Stiffness();
  Eigen::Matrix2d E;
  E << 0.1, 0.05, 0.05, 0.05;
  mat.Stress(E).eval();
  }
}

// Register the function as a benchmark
BENCHMARK(BM_isolinear_eval);

//~~~~~~~~~~~~~~~~

// Define another benchmark
static void BM_isolinear_no_opt(benchmark::State &state) {
  mini::IsotropicLinear<double, 2> mat(200, 0.3);
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(mat.Stiffness());
    Eigen::Matrix2d E;
    E << 0.1, 0.05, 0.05, 0.05;
    benchmark::DoNotOptimize(mat.Stress(E).eval());
    }
  }
BENCHMARK(BM_isolinear_no_opt);

BENCHMARK_MAIN();
