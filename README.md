# Lightweight C++ MLP Inference Engine

A small zero-dependency C++17 inference runtime for multilayer perceptrons.

The project focuses on:

- row-major model storage
- cache-friendly dense layer inference
- fixed runtime work buffers
- int8 quantized inference
- simple benchmarks and tests

## What It Supports

- dense MLP layers
- ReLU and linear activations
- float32 inference
- int8 weight and activation quantization
- no heap allocation inside `infer()`
- plain text model loading
- Makefile and CMake builds

It does not train models. The runtime is intentionally small so the low-level inference path is easy to read.

## Layout

```text
include/mlp/       public headers
src/               runtime, demo, and benchmark code
tests/             no-framework C++ tests
models/            sample model file
tools/             helper script for regenerating the sample model
```

## Build

Using Make:

```bash
make demo
make test
make bench
```

Using CMake:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Run

Demo:

```bash
build/mlp_demo
```

Example output:

```text
model: models/tiny_mlp.txt
float parameter bytes: 204
int8 parameter bytes: 86
float workspace bytes: 48
int8 workspace bytes: 36
float output: 0.94400 -0.57400 0.29600
int8 output: 0.94538 -0.57614 0.29612
```

Benchmark:

```bash
build/mlp_benchmark models/tiny_mlp.txt 200000
```

Example result on a local Fedora machine:

```text
runtime                   avg us          checksum
float naive               0.0300       44399.92657
float optimized           0.0294       44399.92657
int8 quantized            0.0919       44357.00295
```

The quantized path is smaller in memory, but slower on this tiny model because dynamic activation quantization overhead is larger than the matrix multiply itself. Larger models make the compute path more meaningful.

## Model Format

The sample model uses a simple text format:

```text
layers 2
layer 4 6 relu
weights
...
bias
...
layer 6 3 linear
weights
...
bias
...
```

Weights are stored in row-major order:

```text
weight_index = output_neuron * input_size + input_neuron
```

This lets each output neuron read one contiguous row of weights during matrix-vector multiplication.

## Runtime Design

The float runtime has two kernels:

- `FloatKernel::Naive`: simple reference implementation
- `FloatKernel::Optimized`: pointer-based row-major loop with small unrolling

The quantized runtime:

- converts float weights to int8 once during construction
- dynamically quantizes activations per layer
- accumulates products in int32
- dequantizes layer output back to float

Both runtimes allocate workspace buffers in the constructor. During inference, callers pass an input vector and a correctly sized output vector.

## Tests

```bash
make test
```

Current checks cover:

- dense layer math
- ReLU behavior
- model metadata
- model loading
- naive vs optimized output matching
- quantization helper behavior
- float vs int8 output tolerance
- output size validation

## Resume Bullets

- Built a zero-dependency C++17 inference runtime for MLP models with row-major weight storage and fixed workspace buffers.
- Implemented float32 and int8 inference paths using int32 accumulation and dynamic activation quantization.
- Reduced sample model parameter storage from 204 bytes to 86 bytes with int8 quantization while keeping output error within test tolerance.

## Future Extensions

- AVX2 or NEON kernel behind a compile-time flag
- OpenMP batching for multiple input samples
- fixed calibration data for faster int8 activation scaling
- `perf` and `valgrind/cachegrind` profiling notes
