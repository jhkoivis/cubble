#include <cuda.h>
#include <math.h>
#include <stdlib.h>

#include "cubble_tests.cuh"
#include "Kernels.cuh"
#include "gtest/gtest.h"

// Google test documention can be found in: https://github.com/google/googletest/tree/master/googletest

// Here TEST functions can be built
// TEST function takes two arguments, which can be named anything. First which TestSuite the Test belongs to. Tests that
// test similar functionality should have the same test suite name. Secondly the test name. Should be descriptive, since
// this will be shown, if Test fails
// ---------------------------------------------------------------------------------------------------------------------
// Below are two examples of how tests can be used with cubble
// ---------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------
// Tests if a simple cubble vector addition function works. The function adds the values of
// arrays together
// ------------------------------------------------------------------------------------------

// Just forwards to cubble function, since cannot call __device__ function from host
template<typename ...Args>
__global__ void TestGtestWorkingSuite_AddArrays(Args... args) {cubble::test_vec_add(args...);}

TEST(TestGtestWorkingSuite, AddArrays){
    /*
     * Initialise two arrays with one million numbers, add together and then test, if correctly added
     */
	int N = 1e6;
	float *x, *y, *z;
	float *x_d, *y_d, *z_d;
	double error_tolerance = 1e-5;  // Error tolerance for comparison

	// Reserve memory for arrays on host
	x = (float *)malloc(sizeof(float) * N);
	y = (float *)malloc(sizeof(float) * N);
	z = (float *)malloc(sizeof(float) * N);

    // Reserve memory for arrays on device
	cudaMalloc( (void **)&x_d, sizeof(float) * N);
	cudaMalloc( (void **)&y_d, sizeof(float) * N);
	cudaMalloc( (void **)&z_d , sizeof(float) * N);

	// initialize x and y arrays
	for (int i = 0; i < N; i++) {
		x[i] = sin(i);
		y[i] = cos(i);
	}

	// Copy x, y values to device
	cudaMemcpy(x_d, x, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Grid and block dimensions
	dim3 gridDim((int)ceil(N / 256.0));
	dim3 blockDim(256);

	// Calls __global__ function that will call correct cubble function
	TestGtestWorkingSuite_AddArrays<<<gridDim, blockDim>>>(x_d, y_d, z_d, N);

	// Wait for all threads to finish
	cudaDeviceSynchronize();

	// Copy result to host
	cudaMemcpy(z, z_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

	// Test every result against expected result within error tolerance. If anything is not within the tolerance the
	// entire Test will fail
	for(int i = 0; i < N; i++){
		EXPECT_NEAR(x[i] + y[i], z[i], error_tolerance);
	}

	// Free the memory

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

// Forwards call to cubble. Each thread executes M steps, i.e. calls eulerIntegrate M times
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

int N = 1000;  // Number of initial points
int M = 500;  // Number of steps to execute
double initial_conditions[N];  // Array to store the initial points

double 	timeStep = 1.0;  // Euler step size
double 	*y;  // Array to store result
double 	*f;  // Array to store values of f function

// Arrays on device
double 	*y_d;
double 	*f_d;

double error_tolerance = 1e-5;  // Error tolerance for comparison

// Reserve memory on host
y = 		(double *)malloc(sizeof(double) * N);
f = 		(double *)malloc(sizeof(double) * N);

// Reserve memory on device
cudaMalloc( (void **)&y_d, sizeof(double) * N);
cudaMalloc( (void **)&f_d, sizeof(double) * N);

// returns values of function f(y). Here simply f(y) = 1
auto func_linear_derivative = [](double y_f)
{
  return 1.0;
};

// Initialize initial conditions
for (int i = 0; i < N; i++) {
	initial_conditions[i] = rand() % 100;  // Random number as intial condition
	y[i] = initial_conditions[i];
	f[i] = func_linear_derivative(y[i]);  // Calculate associated f values
}

// Copy values to device
cudaMemcpy(f_d, f, sizeof(double) * N, cudaMemcpyHostToDevice);
cudaMemcpy(y_d, y, sizeof(double) * N, cudaMemcpyHostToDevice);

// Grid, Threads
dim3 gridDim((int)ceil(N / 256.0));
dim3 blockDim(256);

// Forward call to __global__ function
TestCubbleKernels_EulerIntegrate<<<gridDim, blockDim>>>(N, M, timeStep, y_d, f_d);

// Wait for all to finish
cudaDeviceSynchronize();

// Copy result to host
cudaMemcpy(y, y_d, sizeof(double) * N, cudaMemcpyDeviceToHost);

// Check each result and compare to expectation (initial value + number of steps * step size)
// If one is outside tolerance entire Test fails
for(int i = 0; i < N; i++){
	EXPECT_NEAR(initial_conditions[i] + M * timeStep, y[i], error_tolerance);
}

// Free the memory

cudaFree(y_d);
cudaFree(f_d);

free(f);
free(y);

}
// ---------------------------------------------------------------------------------------------------------------------