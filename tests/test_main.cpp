#include "mlp/model.hpp"
#include "mlp/quantization.hpp"
#include "mlp/runtime.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, float tolerance, const char* message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

mlp::Model single_layer_model() {
    mlp::Layer layer;
    layer.input_size = 2;
    layer.output_size = 2;
    layer.weights = {
        1.0f, 2.0f,
        -1.0f, 0.5f,
    };
    layer.bias = {0.5f, -0.25f};
    layer.relu = true;

    mlp::Model model;
    model.layers.push_back(std::move(layer));
    return model;
}

void test_float_runtime_math() {
    mlp::FloatRuntime runtime(single_layer_model(), mlp::FloatKernel::Naive);
    std::vector<float> input{3.0f, -1.0f};
    std::vector<float> output(2);

    runtime.infer(input, output);

    require_close(output[0], 1.5f, 0.0001f, "first dense output is wrong");
    require_close(output[1], 0.0f, 0.0001f, "relu output is wrong");
}

void test_model_metadata() {
    mlp::Model model = single_layer_model();

    require(model.input_size() == 2, "input size metadata is wrong");
    require(model.output_size() == 2, "output size metadata is wrong");
    require(model.parameter_count() == 6, "parameter count is wrong");
    require(model.float_bytes() == 24, "float byte count is wrong");
}

void test_model_loader() {
    mlp::Model model = mlp::load_model("models/tiny_mlp.txt");

    require(model.layers.size() == 2, "tiny model layer count is wrong");
    require(model.input_size() == 4, "tiny model input size is wrong");
    require(model.output_size() == 3, "tiny model output size is wrong");
    require(model.parameter_count() == 51, "tiny model parameter count is wrong");
}

void test_naive_and_optimized_match() {
    mlp::Model model = mlp::load_model("models/tiny_mlp.txt");
    mlp::FloatRuntime naive(model, mlp::FloatKernel::Naive);
    mlp::FloatRuntime optimized(model, mlp::FloatKernel::Optimized);

    std::vector<float> input{0.6f, -0.3f, 0.8f, 0.2f};
    std::vector<float> naive_output(3);
    std::vector<float> optimized_output(3);

    naive.infer(input, naive_output);
    optimized.infer(input, optimized_output);

    for (std::size_t i = 0; i < naive_output.size(); ++i) {
        require_close(naive_output[i], optimized_output[i], 0.0001f, "optimized output mismatch");
    }
}

void test_quantization_helpers() {
    const std::vector<float> values{-1.0f, 0.0f, 1.0f};
    const float scale = mlp::choose_scale(values.data(), static_cast<int>(values.size()));

    require_close(scale, 1.0f / 127.0f, 0.000001f, "scale choice is wrong");
    require(mlp::quantize_to_int8(1.0f, scale) == 127, "positive quantization is wrong");
    require(mlp::quantize_to_int8(-1.0f, scale) == -127, "negative quantization is wrong");
}

void test_quantized_runtime_close_to_float() {
    mlp::Model model = mlp::load_model("models/tiny_mlp.txt");
    mlp::FloatRuntime float_runtime(model);
    mlp::QuantizedRuntime quantized_runtime(model);

    std::vector<float> input{0.6f, -0.3f, 0.8f, 0.2f};
    std::vector<float> float_output(3);
    std::vector<float> quantized_output(3);

    float_runtime.infer(input, float_output);
    quantized_runtime.infer(input, quantized_output);

    for (std::size_t i = 0; i < float_output.size(); ++i) {
        require_close(float_output[i], quantized_output[i], 0.01f, "quantized output drift is too large");
    }
}

void test_output_size_validation() {
    mlp::FloatRuntime runtime(single_layer_model());
    std::vector<float> input{1.0f, 1.0f};
    std::vector<float> output(1);

    bool threw = false;
    try {
        runtime.infer(input, output);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    require(threw, "runtime should reject wrong output size");
}

}  // namespace

int main() {
    try {
        test_float_runtime_math();
        test_model_metadata();
        test_model_loader();
        test_naive_and_optimized_match();
        test_quantization_helpers();
        test_quantized_runtime_close_to_float();
        test_output_size_validation();
    } catch (const std::exception& error) {
        std::cerr << "test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "all tests passed\n";
    return 0;
}
