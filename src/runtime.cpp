#include "mlp/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlp {
namespace {

std::size_t max_width(const Model& model) {
    std::size_t width = static_cast<std::size_t>(model.input_size());
    for (const Layer& layer : model.layers) {
        width = std::max(width, static_cast<std::size_t>(layer.output_size));
    }
    return width;
}

void dense_layer_naive(const Layer& layer, const float* input, float* output) {
    for (int row = 0; row < layer.output_size; ++row) {
        float sum = layer.bias[static_cast<std::size_t>(row)];
        for (int col = 0; col < layer.input_size; ++col) {
            const auto weight_index = static_cast<std::size_t>(row) *
                                      static_cast<std::size_t>(layer.input_size) +
                                      static_cast<std::size_t>(col);
            sum += layer.weights[weight_index] * input[col];
        }
        output[row] = layer.relu ? std::max(0.0f, sum) : sum;
    }
}

void dense_layer_optimized(const Layer& layer, const float* input, float* output) {
    const int input_size = layer.input_size;
    const float* weights = layer.weights.data();

    for (int row = 0; row < layer.output_size; ++row) {
        const float* row_weights = weights + static_cast<std::size_t>(row) *
                                             static_cast<std::size_t>(input_size);
        float sum = layer.bias[static_cast<std::size_t>(row)];

        int col = 0;
        for (; col + 3 < input_size; col += 4) {
            sum += row_weights[col] * input[col];
            sum += row_weights[col + 1] * input[col + 1];
            sum += row_weights[col + 2] * input[col + 2];
            sum += row_weights[col + 3] * input[col + 3];
        }
        for (; col < input_size; ++col) {
            sum += row_weights[col] * input[col];
        }

        output[row] = layer.relu ? std::max(0.0f, sum) : sum;
    }
}

}  // namespace

FloatRuntime::FloatRuntime(Model model, FloatKernel kernel)
    : model_(std::move(model)), kernel_(kernel) {
    validate_model(model_);
    const std::size_t width = max_width(model_);
    buffer_a_.resize(width);
    buffer_b_.resize(width);
}

void FloatRuntime::infer(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != static_cast<std::size_t>(model_.input_size())) {
        throw std::runtime_error("input size does not match model");
    }
    if (output.size() != static_cast<std::size_t>(model_.output_size())) {
        throw std::runtime_error("output size does not match model");
    }

    const float* current_input = input.data();
    float* current_output = buffer_a_.data();

    for (std::size_t i = 0; i < model_.layers.size(); ++i) {
        const Layer& layer = model_.layers[i];
        current_output = (i % 2 == 0) ? buffer_a_.data() : buffer_b_.data();
        if (kernel_ == FloatKernel::Naive) {
            dense_layer_naive(layer, current_input, current_output);
        } else {
            dense_layer_optimized(layer, current_input, current_output);
        }
        current_input = current_output;
    }

    const int final_size = model_.output_size();
    for (int i = 0; i < final_size; ++i) {
        output[static_cast<std::size_t>(i)] = current_input[i];
    }
}

std::size_t FloatRuntime::workspace_bytes() const {
    return (buffer_a_.size() + buffer_b_.size()) * sizeof(float);
}

const Model& FloatRuntime::model() const {
    return model_;
}

}  // namespace mlp
