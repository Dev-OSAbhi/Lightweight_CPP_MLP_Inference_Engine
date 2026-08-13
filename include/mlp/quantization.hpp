#pragma once

#include "mlp/model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlp {

struct QuantizedLayer {
    int input_size = 0;
    int output_size = 0;
    std::vector<std::int8_t> weights;
    std::vector<float> bias;
    float weight_scale = 1.0f;
    bool relu = false;
};

struct QuantizedModel {
    std::vector<QuantizedLayer> layers;

    int input_size() const;
    int output_size() const;
    std::size_t parameter_count() const;
    std::size_t storage_bytes() const;
};

float choose_scale(const float* values, int size);
std::int8_t quantize_to_int8(float value, float scale);
void quantize_values(const float* input, int size, float scale, std::int8_t* output);
QuantizedModel quantize_model(const Model& model);

class QuantizedRuntime {
public:
    explicit QuantizedRuntime(const Model& model);

    void infer(const std::vector<float>& input, std::vector<float>& output);
    std::size_t workspace_bytes() const;
    const QuantizedModel& model() const;

private:
    QuantizedModel model_;
    std::vector<std::int8_t> buffer_a_;
    std::vector<std::int8_t> buffer_b_;
    std::vector<float> float_buffer_;
};

}  // namespace mlp
