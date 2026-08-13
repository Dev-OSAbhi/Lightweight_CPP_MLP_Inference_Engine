#pragma once

#include "mlp/model.hpp"

#include <cstddef>
#include <vector>

namespace mlp {

enum class FloatKernel {
    Naive,
    Optimized,
};

class FloatRuntime {
public:
    explicit FloatRuntime(Model model, FloatKernel kernel = FloatKernel::Optimized);

    void infer(const std::vector<float>& input, std::vector<float>& output);
    std::size_t workspace_bytes() const;
    const Model& model() const;

private:
    Model model_;
    FloatKernel kernel_ = FloatKernel::Optimized;
    std::vector<float> buffer_a_;
    std::vector<float> buffer_b_;
};

}  // namespace mlp
