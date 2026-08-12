#include "mlp/model.hpp"

#include <fstream>
#include <stdexcept>

namespace mlp {
namespace {

void expect_token(std::istream& input, const std::string& expected) {
    std::string actual;
    if (!(input >> actual) || actual != expected) {
        throw std::runtime_error("expected token: " + expected);
    }
}

Layer read_layer(std::istream& input) {
    expect_token(input, "layer");

    Layer layer;
    std::string activation;
    if (!(input >> layer.input_size >> layer.output_size >> activation)) {
        throw std::runtime_error("invalid layer header");
    }
    if (layer.input_size <= 0 || layer.output_size <= 0) {
        throw std::runtime_error("layer sizes must be positive");
    }
    if (activation == "relu") {
        layer.relu = true;
    } else if (activation == "linear") {
        layer.relu = false;
    } else {
        throw std::runtime_error("unknown activation: " + activation);
    }

    expect_token(input, "weights");
    layer.weights.resize(static_cast<std::size_t>(layer.input_size) *
                         static_cast<std::size_t>(layer.output_size));
    for (float& value : layer.weights) {
        if (!(input >> value)) {
            throw std::runtime_error("invalid layer weights");
        }
    }

    expect_token(input, "bias");
    layer.bias.resize(static_cast<std::size_t>(layer.output_size));
    for (float& value : layer.bias) {
        if (!(input >> value)) {
            throw std::runtime_error("invalid layer bias");
        }
    }

    return layer;
}

}  // namespace

int Model::input_size() const {
    return layers.empty() ? 0 : layers.front().input_size;
}

int Model::output_size() const {
    return layers.empty() ? 0 : layers.back().output_size;
}

std::size_t Model::parameter_count() const {
    std::size_t total = 0;
    for (const Layer& layer : layers) {
        total += layer.weights.size();
        total += layer.bias.size();
    }
    return total;
}

std::size_t Model::float_bytes() const {
    return parameter_count() * sizeof(float);
}

Model load_model(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open model: " + path);
    }

    expect_token(input, "layers");

    int layer_count = 0;
    if (!(input >> layer_count) || layer_count <= 0) {
        throw std::runtime_error("invalid layer count");
    }

    Model model;
    model.layers.reserve(static_cast<std::size_t>(layer_count));
    for (int i = 0; i < layer_count; ++i) {
        model.layers.push_back(read_layer(input));
    }

    validate_model(model);
    return model;
}

void validate_model(const Model& model) {
    if (model.layers.empty()) {
        throw std::runtime_error("model must contain at least one layer");
    }

    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const Layer& layer = model.layers[i];
        const auto expected_weights = static_cast<std::size_t>(layer.input_size) *
                                      static_cast<std::size_t>(layer.output_size);
        if (layer.input_size <= 0 || layer.output_size <= 0) {
            throw std::runtime_error("layer sizes must be positive");
        }
        if (layer.weights.size() != expected_weights) {
            throw std::runtime_error("layer weight size mismatch");
        }
        if (layer.bias.size() != static_cast<std::size_t>(layer.output_size)) {
            throw std::runtime_error("layer bias size mismatch");
        }
        if (i > 0 && model.layers[i - 1].output_size != layer.input_size) {
            throw std::runtime_error("adjacent layer sizes do not match");
        }
    }
}

}  // namespace mlp
