#include <cuda.h>
#include <math.h>
#include <stdlib.h>

#include "cubble_tests.cuh"
#include "Kernels.cuh"
#include "gtest/gtest.h"



// ------------------------------------------------------------------------------------------
// Tests if a simple cubble vector addition function works. The function adds the values of
// arrays together
// ------------------------------------------------------------------------------------------
template<typename ...Args>
__global__ void TestGtestWorkingSuite_AddArrays(Args... args) {cubble::test_vec_add(args...);}

TEST(TestGtestWorkingSuite, AddArrays){

	int N = 1e6;
	float *x, *y, *z;
	float *x_d, *y_d, *z_d;
	double error_tolerance = 1e-5;

	x = (float *)malloc(sizeof(float) * N);
	y = (float *)malloc(sizeof(float) * N);
	z = (float *)malloc(sizeof(float) * N);

	cudaMalloc( (void **)&x_d, sizeof(float) * N);
	cudaMalloc( (void **)&y_d, sizeof(float) * N);
	cudaMalloc( (void **)&z_d , sizeof(float) * N);

	// initialize x and y arrays
	for (int i = 0; i < N; i++) {
		x[i] = sin(i);
		y[i] = cos(i);
	}

	cudaMemcpy(x_d, x, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float) * N, cudaMemcpyHostToDevice);

	dim3 gridDim((int)ceil(N / 256.0));
	dim3 blockDim(256);

	TestGtestWorkingSuite_AddArrays<<<gridDim, blockDim>>>(x_d, y_d, z_d, N);

	cudaDeviceSynchronize();

	cudaMemcpy(z, z_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++){
		EXPECT_NEAR(x[i] + y[i], z[i], error_tolerance);
	}

	// Free memory

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);

	free(x);
	free(y);
	free(z);

}


// ------------------------------------------------------------------------------------------
// Tests if the euler method kernel is working correctly. This is done by assuming the ODE: y' = f(y) = 1
// This is effectively a line, i.e y(t) = t + a, with a the initial condition
// E.g. Using the Euler method for the initial condition y(0) = 5 with 3 steps of size 2 yields y(6) = 11.
// Here random initial conditions are generated for N ys. Each thread propagates one y element M times.
// At the end the propagated ys are compared to the expect analytical values
// ------------------------------------------------------------------------------------------

template<typename ...Args>
__global__ void TestCubbleKernels_EulerIntegrate(int N, int M, Args... args) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N){
		for (int i = 0; i < M; ++i){
			cubble::eulerIntegrate(tid, args...);
		}
	}
}


TEST(TestCubbleKernels, EulerIntegrate){

int N = 1000;
int M = 500;
double initial_conditions[N];

double 	timeStep = 1.0;
double 	*y;
double 	*f;

double 	*y_d;
double 	*f_d;

double error_tolerance = 1e-5;

y = 		(double *)malloc(sizeof(double) * N);
f = 		(double *)malloc(sizeof(double) * N);

cudaMalloc( (void **)&y_d, sizeof(double) * N);
cudaMalloc( (void **)&f_d, sizeof(double) * N);

auto func_linear_derivative = [](double y_f)
{
  return 1.0;
};

// Initialize initial conditions
for (int i = 0; i < N; i++) {
	initial_conditions[i] = rand() % 100;
	y[i] = initial_conditions[i];
	f[i] = func_linear_derivative(y[i]);
}

cudaMemcpy(f_d, f, sizeof(double) * N, cudaMemcpyHostToDevice);
cudaMemcpy(y_d, y, sizeof(double) * N, cudaMemcpyHostToDevice);

dim3 gridDim((int)ceil(N / 256.0));
dim3 blockDim(256);

TestCubbleKernels_EulerIntegrate<<<gridDim, blockDim>>>(N, M, timeStep, y_d, f_d);

cudaDeviceSynchronize();

cudaMemcpy(y, y_d, sizeof(double) * N, cudaMemcpyDeviceToHost);


for(int i = 0; i < N; i++){
	EXPECT_NEAR(initial_conditions[i] + M * timeStep, y[i], error_tolerance);
}

// Free memory

cudaFree(y_d);
cudaFree(f_d);

free(f);
free(y);

}