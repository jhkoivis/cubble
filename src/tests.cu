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
	cout << EQUAL(1, 1) << endl;
}

void TEST_simple_cuda (void)
{
	int N = 128*16;
	int *x, *y, *z;
	int *x_d, *y_d, *z_d;

	x = (int *)malloc(sizeof(int) * N);
	y = (int *)malloc(sizeof(int) * N);
	z = (int *)malloc(sizeof(int) * N);
	cudaMalloc( (void **)&x_d, sizeof(int) * N);
	cudaMalloc( (void **)&y_d, sizeof(int) * N);
	cudaMalloc( (void **)&z_d , sizeof(int) * N);

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1+i;
		y[i] = 2;
	}

	cudaMemcpy(x_d, x, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(int) * N, cudaMemcpyHostToDevice);

	// Run kernel on 1M elements on the GPU
	cuda_test<<<16, 128>>>(x_d, y_d, z_d);
	cudaDeviceSynchronize();

	cudaMemcpy(z, z_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++){
		cout << "z = " << z[i] << " for i = " << i << endl;
	}

	// Free memory

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);

	free(x);
	free(y);
	free(z);
}

};

/*
__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f)
{
	y[idx] += f[idx] * timeStep;
}*/
