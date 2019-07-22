#include "Util.h"
#include <cuda.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tests.cuh"
#include "Kernels.cuh"

using namespace std;
using namespace cubble;

namespace ASSERT{
  int EQUAL(int x, int y){
	  if (x == y){
	  	cout << "Passed" << endl;
		  return 1;
	  }
	  else{
	  	cout << "Not passed" << endl;
		  return 0;
	  }
  }
};

namespace TestSuite{
using namespace ASSERT;

void TEST_tests_are_running (void)
{
	cout << "Tests ready to start" << endl;
}

void TEST_cubble_simple (void)
{
	cout << test_me() << endl;
	cout << EQUAL(1, 2) << endl;
}

void TEST_simple_kernel (void)
{
	int N = 10;
	double *f, *y;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&f, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		f[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the GPU
	eulerIntegrate<<<1, 1>>>(0, 1.0, y, f);

	cout << y[0] << endl;

	// Free memory
	cudaFree(f);
	cudaFree(y);
}

};

/*
__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f)
{
	y[idx] += f[idx] * timeStep;
}*/
