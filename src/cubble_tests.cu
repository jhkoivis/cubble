#include "Util.h"
#include <cuda.h>
#include <math.h>

#include "cubble_tests.cuh"
#include "Kernels.cuh"
#include "gtest/gtest.h"


__global__ void test_vec_add_cubble(float* x, float* y, float* z, int N){

	cubble::test_vec_add(x, y, z, N);

}


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

	test_vec_add_cubble<<<gridDim, blockDim>>>(x_d, y_d, z_d, N);

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
