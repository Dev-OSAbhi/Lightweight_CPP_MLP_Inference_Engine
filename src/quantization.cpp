#include "mlp/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlp {
namespace {

std::size_t max_width(const QuantizedModel& model) {
    std::size_t width = static_cast<std::size_t>(model.input_size());
    for (const QuantizedLayer& layer : model.layers) {
        width = std::max(width, static_cast<std::size_t>(layer.output_size));
    }
    return width;
}

void validate_quantized_model(const QuantizedModel& model) {
    if (model.layers.empty()) {
        throw std::runtime_error("quantized model must contain at least one layer");
    }

    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const QuantizedLayer& layer = model.layers[i];
        const auto expected_weights = static_cast<std::size_t>(layer.input_size) *
                                      static_cast<std::size_t>(layer.output_size);
        if (layer.input_size <= 0 || layer.output_size <= 0) {
            throw std::runtime_error("quantized layer sizes must be positive");
        }
        if (layer.weights.size() != expected_weights) {
            throw std::runtime_error("quantized layer weight size mismatch");
        }
        if (layer.bias.size() != static_cast<std::size_t>(layer.output_size)) {
            throw std::runtime_error("quantized layer bias size mismatch");
        }
        if (layer.weight_scale <= 0.0f) {
            throw std::runtime_error("quantized layer scale must be positive");
        }
        if (i > 0 && model.layers[i - 1].output_size != layer.input_size) {
            throw std::runtime_error("adjacent quantized layer sizes do not match");
        }
    }
}

void run_quantized_layer(const QuantizedLayer& layer,
                         const std::int8_t* input,
                         float input_scale,
                         float* output) {
    const int input_size = layer.input_size;
    const std::int8_t* weights = layer.weights.data();
    const float combined_scale = input_scale * layer.weight_scale;

    for (int row = 0; row < layer.output_size; ++row) {
        const std::int8_t* row_weights = weights + static_cast<std::size_t>(row) *
                                                   static_cast<std::size_t>(input_size);
        std::int32_t acc = 0;

        int col = 0;
        for (; col + 3 < input_size; col += 4) {
            acc += static_cast<std::int32_t>(row_weights[col]) * input[col];
            acc += static_cast<std::int32_t>(row_weights[col + 1]) * input[col + 1];
            acc += static_cast<std::int32_t>(row_weights[col + 2]) * input[col + 2];
            acc += static_cast<std::int32_t>(row_weights[col + 3]) * input[col + 3];
        }
        for (; col < input_size; ++col) {
            acc += static_cast<std::int32_t>(row_weights[col]) * input[col];
        }

        float value = static_cast<float>(acc) * combined_scale + layer.bias[row];
        output[row] = layer.relu ? std::max(0.0f, value) : value;
    }
}

}  // namespace

int QuantizedModel::input_size() const {
    return layers.empty() ? 0 : layers.front().input_size;
}

int QuantizedModel::output_size() const {
    return layers.empty() ? 0 : layers.back().output_size;
}

std::size_t QuantizedModel::parameter_count() const {
    std::size_t total = 0;
    for (const QuantizedLayer& layer : layers) {
        total += layer.weights.size();
        total += layer.bias.size();
    }
    return total;
}

std::size_t QuantizedModel::storage_bytes() const {
    std::size_t total = 0;
    for (const QuantizedLayer& layer : layers) {
        total += layer.weights.size() * sizeof(std::int8_t);
        total += layer.bias.size() * sizeof(float);
        total += sizeof(float);
    }
    return total;
}

float choose_scale(const float* values, int size) {
    if (size <= 0) {
        throw std::runtime_error("cannot choose scale for empty values");
    }

    float max_abs = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_abs = std::max(max_abs, std::fabs(values[i]));
    }

    return max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
}

std::int8_t quantize_to_int8(float value, float scale) {
    if (scale <= 0.0f) {
        throw std::runtime_error("quantization scale must be positive");
    }

    const int rounded = static_cast<int>(std::lround(value / scale));
    const int clipped = std::max(-127, std::min(127, rounded));
    return static_cast<std::int8_t>(clipped);
}

void quantize_values(const float* input, int size, float scale, std::int8_t* output) {
    for (int i = 0; i < size; ++i) {
        output[i] = quantize_to_int8(input[i], scale);
    }
}

QuantizedModel quantize_model(const Model& model) {
    validate_model(model);

    QuantizedModel quantized;
    quantized.layers.reserve(model.layers.size());

    for (const Layer& layer : model.layers) {
        QuantizedLayer qlayer;
        qlayer.input_size = layer.input_size;
        qlayer.output_size = layer.output_size;
        qlayer.bias = layer.bias;
        qlayer.relu = layer.relu;
        qlayer.weight_scale = choose_scale(layer.weights.data(),
                                           static_cast<int>(layer.weights.size()));
        qlayer.weights.resize(layer.weights.size());
        quantize_values(layer.weights.data(),
                        static_cast<int>(layer.weights.size()),
                        qlayer.weight_scale,
                        qlayer.weights.data());
        quantized.layers.push_back(std::move(qlayer));
    }

    validate_quantized_model(quantized);
    return quantized;
}

QuantizedRuntime::QuantizedRuntime(const Model& model) : model_(quantize_model(model)) {
    const std::size_t width = max_width(model_);
    buffer_a_.resize(width);
    buffer_b_.resize(width);
    float_buffer_.resize(width);
}

void QuantizedRuntime::infer(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != static_cast<std::size_t>(model_.input_size())) {
        throw std::runtime_error("input size does not match quantized model");
    }
    if (output.size() != static_cast<std::size_t>(model_.output_size())) {
        throw std::runtime_error("output size does not match quantized model");
    }

    float input_scale = choose_scale(input.data(), static_cast<int>(input.size()));
    quantize_values(input.data(), static_cast<int>(input.size()), input_scale, buffer_a_.data());

    const std::int8_t* current_input = buffer_a_.data();
    std::int8_t* current_quantized_output = buffer_b_.data();

    for (std::size_t i = 0; i < model_.layers.size(); ++i) {
        const QuantizedLayer& layer = model_.layers[i];
        run_quantized_layer(layer, current_input, input_scale, float_buffer_.data());

        const bool is_last = i + 1 == model_.layers.size();
        if (is_last) {
            for (int j = 0; j < layer.output_size; ++j) {
                output[static_cast<std::size_t>(j)] = float_buffer_[j];
            }
        } else {
            input_scale = choose_scale(float_buffer_.data(), layer.output_size);
            current_quantized_output = (i % 2 == 0) ? buffer_b_.data() : buffer_a_.data();
            quantize_values(float_buffer_.data(), layer.output_size, input_scale, current_quantized_output);
            current_input = current_quantized_output;
        }
    }
}

std::size_t QuantizedRuntime::workspace_bytes() const {
    return buffer_a_.size() * sizeof(std::int8_t) +
           buffer_b_.size() * sizeof(std::int8_t) +
           float_buffer_.size() * sizeof(float);
}

const QuantizedModel& QuantizedRuntime::model() const {
    return model_;
}

}  // namespace mlp
