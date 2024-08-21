
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <random>


// CUDA kernel to calculate Black-Scholes formula and average stock price
__global__ void blackScholesKernel(float* prices, float* strikes, float* times, float* risks, float* vols, float* results, int numOptions, int numSims, float* avgPrice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numOptions) {
        float sum = 0.0f;
        for (int j = 0; j < numSims; j++) {
            // Generate random numbers on GPU
            curandState rng;
            curand_init(i * numSims + j, 0, 0, &rng);
            float Z = sqrt(-2.0 * log(curand_uniform(&rng))) * cos(2.0 * 3.1416 * curand_uniform(&rng));
            float St = prices[i] * exp((risks[i] - 0.5 * vols[i] * vols[i]) * times[i] + vols[i] * sqrt(times[i]) * Z);
            float payoff = fmax(St - strikes[i], 0.0f);
            sum += payoff;
        }
        results[i] = exp(-risks[i] * times[i]) * sum / numSims;

        // Calculate average stock price
        atomicAdd(avgPrice, prices[i]);
    }
}

int main() {

    // Initialize input data
    int numOptions = 10000;
    int numSims = 100;
    float* prices;
    float* strikes;
    float* times;
    float* risks;
    float* vols;
    float* results;
    cudaMallocManaged(&prices, numOptions * sizeof(float));
    cudaMallocManaged(&strikes, numOptions * sizeof(float));
    cudaMallocManaged(&times, numOptions * sizeof(float));
    cudaMallocManaged(&risks, numOptions * sizeof(float));
    cudaMallocManaged(&vols, numOptions * sizeof(float));
    cudaMallocManaged(&results, numOptions * sizeof(float));
    float* avgPrice;
    cudaMallocManaged(&avgPrice, sizeof(float));
    *avgPrice = 0.0f;

    // Initialize input data arrays with random values
    for (int i = 0; i < numOptions; i++) {
        prices[i] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
        strikes[i] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
        times[i] = 5.0f * static_cast<float>(rand()) / RAND_MAX;
        risks[i] = 0.1f * static_cast<float>(rand()) / RAND_MAX;
        vols[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numOptions + threadsPerBlock - 1) / threadsPerBlock;
    blackScholesKernel << <blocksPerGrid, threadsPerBlock >> > (prices, strikes, times, risks, vols, results, numOptions, numSims, avgPrice);
    cudaDeviceSynchronize();

    // Calculate average stock price
    *avgPrice /= numOptions;
    std::cout << "Average stock price: " << *avgPrice << std::endl;

    // Print results
    for (int i = 0; i < numOptions; i++) {
        std::cout << "Option " << i << ": " << results[i] << std::endl;
    }

    // Clean up
    cudaFree(prices);
    cudaFree(strikes);
    cudaFree(times);
    cudaFree(risks);
    cudaFree(vols);
    cudaFree(results);
    cudaFree(avgPrice);

    return 0;
}
