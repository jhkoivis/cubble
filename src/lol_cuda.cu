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


void simple(int x){
	printf("%d", x);
}

__device__ void dv(float* x, float* y, float* z, int N){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N){
		z[tid] = x[tid] + y[tid];
	}
}

__global__ void test_local(float* x, float* y, float* z, int N){

	cubble::ctst(x, y, z, N);

}


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

void tests_are_running (void)
{
	cout << "Tests ready to start" << endl;
}

void cubble_simple (void)
{
	cout << test_me() << endl;
	cout << EQUAL(1, 1) << endl;
}

void simple_cuda (void)
{
	int N = 32;
	float *x, *y, *z;
	float *x_d, *y_d, *z_d;

	x = (float *)malloc(sizeof(float) * N);
	y = (float *)malloc(sizeof(float) * N);
	z = (float *)malloc(sizeof(float) * N);
	cudaMalloc( (void **)&x_d, sizeof(float) * N);
	cudaMalloc( (void **)&y_d, sizeof(float) * N);
	cudaMalloc( (void **)&z_d , sizeof(float) * N);

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1+i;
		y[i] = 2;
	}

	cudaMemcpy(x_d, x, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float) * N, cudaMemcpyHostToDevice);

	dim3 gridDim(1);
	dim3 blockDim(N);

	// Run kernel on 1M elements on the GPU
	test_local<<<gridDim, blockDim>>>(x_d, y_d, z_d, N);

	cudaDeviceSynchronize();

	cudaMemcpy(z, z_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

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
