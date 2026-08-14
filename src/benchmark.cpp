#include "mlp/model.hpp"
#include "mlp/quantization.hpp"
#include "mlp/runtime.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct BenchResult {
    double average_us = 0.0;
    double checksum = 0.0;
};

template <typename Runtime>
BenchResult run_benchmark(Runtime& runtime,
                          const std::vector<float>& input,
                          std::vector<float>& output,
                          int iterations) {
    for (int i = 0; i < 100; ++i) {
        runtime.infer(input, output);
    }

    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        runtime.infer(input, output);
        checksum += output[static_cast<std::size_t>(i) % output.size()];
    }
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    BenchResult result;
    result.average_us = static_cast<double>(elapsed_ns) / 1000.0 / iterations;
    result.checksum = checksum;
    return result;
}

void print_result(const std::string& name, const BenchResult& result) {
    std::cout << std::left << std::setw(18) << name
              << std::right << std::setw(14) << std::fixed << std::setprecision(4)
              << result.average_us
              << std::setw(18) << std::setprecision(5) << result.checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/tiny_mlp.txt";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 200000;

    if (iterations <= 0) {
        std::cerr << "error: iterations must be positive\n";
        return 1;
    }

    try {
        mlp::Model model = mlp::load_model(model_path);
        mlp::FloatRuntime naive(model, mlp::FloatKernel::Naive);
        mlp::FloatRuntime optimized(model, mlp::FloatKernel::Optimized);
        mlp::QuantizedRuntime quantized(model);

        std::vector<float> input{0.6f, -0.3f, 0.8f, 0.2f};
        std::vector<float> output(static_cast<std::size_t>(model.output_size()));

        const BenchResult naive_result = run_benchmark(naive, input, output, iterations);
        const BenchResult optimized_result = run_benchmark(optimized, input, output, iterations);
        const BenchResult quantized_result = run_benchmark(quantized, input, output, iterations);

        std::cout << "model: " << model_path << '\n';
        std::cout << "iterations: " << iterations << '\n';
        std::cout << "float parameter bytes: " << model.float_bytes() << '\n';
        std::cout << "int8 parameter bytes: " << quantized.model().storage_bytes() << '\n';
        std::cout << '\n';
        std::cout << std::left << std::setw(18) << "runtime"
                  << std::right << std::setw(14) << "avg us"
                  << std::setw(18) << "checksum" << '\n';
        print_result("float naive", naive_result);
        print_result("float optimized", optimized_result);
        print_result("int8 quantized", quantized_result);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
