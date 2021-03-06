
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdio.h>

#define BLOCK_SIZE 16

#define LENGTH 10
#define TIME    5

#define STEP_X 0.5
#define STEP_T 0.1

#define EPSILON 0.001

static double *hostA = nullptr, *hostCurX = nullptr, *hostNextX = nullptr;
static double *devA  = nullptr, *devCurX  = nullptr, *devNextX  = nullptr;

static void _cpuFree() {
	if (::hostA)
		std::free((void *)::hostA);

	if (::hostCurX)
		std::free((void *)::hostCurX);

	if (::hostNextX)
		std::free((void *)::hostNextX);
}

#define cudaCheck
static void _gpuFree() {
	if (::devA)
		cudaCheck(cudaFree((void *)::devA));
	
	if (::devCurX)
		cudaCheck(cudaFree((void *)::devCurX));

	if (::devNextX)
		cudaCheck(cudaFree((void *)::devNextX));
}

/*
* CUDA errors catching block
*/

static void _checkCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#undef cudaCheck
#define cudaCheck(value) _checkCudaErrorAux(__FILE__, __LINE__, #value, value)

static void _checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;

	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;

	system("pause");

	_cpuFree();
	_gpuFree();

	exit(1);
}

/*
 * CUDA kernel
 */

__global__ void kernel(double * __restrict__ matrix, double * __restrict__ xNext, double * __restrict__ xCur,
					   std::size_t size, double a, double stepX, double stepT) {
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < size) {
		double sum = 0.0;
		for (auto i = 0; i < size; i++)
			sum += matrix[idx * size + i] * xCur[i];

		xNext[idx] = xCur[idx] + (1.0 / a) * (xCur[idx] - sum);
	}
}

__global__ void gpuOnlyKernel(double * __restrict__ matrix, double * __restrict__ xNext, double * __restrict__ xCur,
	                          std::size_t size, double a, double stepX, double stepT) {
	auto idx	 = threadIdx.x + blockIdx.x * blockDim.x;
	auto epsilon = 0.0;

	do {
		if (idx < size) {
			double sum = 0.0;
			for (auto i = 0; i < size; i++)
				sum += matrix[idx * size + i] * xCur[i];

			xNext[idx] = xCur[idx] + (1.0 / a) * (xCur[idx] - sum);

			__syncthreads();

			epsilon = 0.0;

			for (auto i = 0; i < size; i++)
				epsilon += fabs(xNext[i] - xCur[i]);

			__syncthreads();

			xCur[idx] = xNext[idx];

			__syncthreads();
		} 
	} while (epsilon > EPSILON);
}

/*
 * Init
 */

static int _cpuInit(std::size_t nPoints, const double stepX, const double stepT) {
	::hostA = (double *)std::calloc(nPoints * nPoints, sizeof(double));
	if (!::hostA)
		return 1;

	::hostCurX = (double *)std::calloc(nPoints, sizeof(double));
	if (!::hostCurX)
		return 1;

	::hostNextX = (double *)std::calloc(nPoints, sizeof(double));
	if (!::hostNextX)
		return 1;
	
	::hostA[0] = ::hostA[nPoints * nPoints - 1] = 1;

	std::unique_ptr<double[]> coefficients(new double[3]);
	coefficients[0] = -stepT / (stepX * stepX);
	coefficients[1] = 1 - 2.0 * coefficients[0];
	coefficients[2] = coefficients[0];

	auto offs = 0;
	for (auto i = 1; i < nPoints - 1; i++) {
		for (auto j = 0; j < 3; j++)
			hostA[i * nPoints + j + offs] = coefficients[j];

		offs++;
	}

	::hostCurX[nPoints - 1]  = 5;
	::hostNextX[nPoints - 1] = 5;

	return 0;
}

static void _gpuInit(size_t nPoints) {
	cudaCheck(cudaMalloc((void **)&::devA, nPoints * nPoints * sizeof(double)));
	cudaCheck(cudaMalloc((void **)&::devCurX, nPoints * sizeof(double)));
	cudaCheck(cudaMalloc((void **)&::devNextX, nPoints * sizeof(double)));

	cudaCheck(cudaMemset((void *)::devA, 0, nPoints * nPoints * sizeof(double)));
	cudaCheck(cudaMemset((void *)::devCurX, 0, nPoints * sizeof(double)));
	cudaCheck(cudaMemset((void *)::devNextX, 0, nPoints * sizeof(double)));
}

/*
 * Helpers
 */

void printToConsole(const double *data, std::size_t size) {
	for (auto i = 0; i < size; i++)
		std::cout << data[i] << " ";
	std::cout << std::endl;
}

/*
 * Main
 */

int main() {
	const auto nPoints = static_cast<std::size_t>(LENGTH / STEP_X);
	const auto time    = static_cast<std::size_t>(TIME / STEP_T);

	const auto stepX = STEP_X;
	const auto stepT = STEP_T;

	if (_cpuInit(nPoints, stepX, stepT)) {
		_cpuFree();
		return 1;
	}

	_gpuInit(nPoints);

	cudaCheck(cudaMemcpy(::devA, ::hostA, nPoints * nPoints * sizeof(double), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(::devCurX, ::hostCurX, nPoints * sizeof(double), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(::devNextX, ::hostNextX, nPoints * sizeof(double), cudaMemcpyHostToDevice));

	// Preparation step

	dim3 nBlocks(1);
	dim3 nThreads(256);

	auto a = hostA[nPoints + 1];

	// Full GPU kernel function

	auto beginTime = std::chrono::steady_clock::now();
	gpuOnlyKernel <<< nBlocks, nThreads >>> (::devA, ::devNextX, ::devCurX, nPoints, hostA[nPoints + 1], stepX, stepT);
	auto gpuOnlyTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - beginTime).count();

	// GPU calculation, CPU stop iteration

	/*auto eps = 0.1;

	while (eps > EPSILON) {
		beginTime = std::chrono::steady_clock::now();
		kernel <<<nBlocks, nThreads>>> (::devA, ::devNextX, ::devCurX, nPoints, hostA[nPoints + 1], stepX, stepT);
		auto gpuAndCpuTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - beginTime);

		cudaCheck(cudaMemcpy(::hostCurX, ::devCurX, nPoints * sizeof(double), cudaMemcpyDeviceToHost));
		cudaCheck(cudaMemcpy(::hostNextX, ::devNextX, nPoints * sizeof(double), cudaMemcpyDeviceToHost));

		for (auto i = 0; i < nPoints; i++) {
			std::cout << ::hostNextX[i] << " ";
			::hostNextX[i] = abs(::hostNextX[i] - ::hostCurX[i]);
		}

		std::cout << std::endl << std::endl;

		eps = std::accumulate(::hostNextX, ::hostNextX + nPoints, 0.0);
		
		cudaCheck(cudaMemcpy(::devCurX, ::devNextX, nPoints * sizeof(double), cudaMemcpyDeviceToDevice));
	}*/

	cudaCheck(cudaMemcpy(::hostCurX, ::devCurX, nPoints * sizeof(double), cudaMemcpyDeviceToHost));


	printToConsole(::hostCurX, nPoints);

	_gpuFree();
	_cpuFree();

	system("pause");

	return 0;
}