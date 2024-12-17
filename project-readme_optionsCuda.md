# Options-Pricing-CUDA

## Slogan
Parallel Power, Simple Config

## Overview

CudaRuntime1 is a high-performance computing application designed to leverage both CPU and GPU resources for intensive computational tasks. The project focuses on efficient options pricing calculations using CUDA parallel processing technology.

## Key Features

### Technical Capabilities
- Black-Scholes option price calculations
- GPU-accelerated parallel processing
- Random number generation across multiple GPU threads
- Synchronization of computational tasks

### Architectural Highlights
- Modular design
- Dynamic configuration management
- Centralized environment-specific settings
- Support for development, testing, and production environments

## Project Structure

- `/src`: Source code files
- `/build`: Compiled binaries and debug information
- `CudaRuntime1.sln`: Visual Studio solution file
- `kernel.cu`: CUDA kernel implementation

## Technologies

- CUDA
- C++
- Microsoft Visual Studio
- x64 architecture
- Parallel computing techniques

## Build Configuration

### Requirements
- NVIDIA CUDA Toolkit
- Microsoft Visual Studio
- x64 platform support

### Compilation
1. Open `CudaRuntime1.sln`
2. Select Debug|x64 configuration
3. Build solution

## Performance

The project leverages GPU acceleration to significantly speed up complex financial computations, particularly options pricing calculations.

## Repository

[GitHub Repository](https://github.com/iCELAKE01/Options-Pricing-CUDA)
