#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mlp {

struct Layer {
    int input_size = 0;
    int output_size = 0;
    std::vector<float> weights;
    std::vector<float> bias;
    bool relu = false;
};

struct Model {
    std::vector<Layer> layers;

    int input_size() const;
    int output_size() const;
    std::size_t parameter_count() const;
    std::size_t float_bytes() const;
};

Model load_model(const std::string& path);
void validate_model(const Model& model);

}  // namespace mlp
