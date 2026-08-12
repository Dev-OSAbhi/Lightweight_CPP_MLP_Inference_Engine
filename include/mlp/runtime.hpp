#pragma once

#include "mlp/model.hpp"

#include <cstddef>
#include <vector>

namespace mlp {

class FloatRuntime {
public:
    explicit FloatRuntime(Model model);

    void infer(const std::vector<float>& input, std::vector<float>& output);
    std::size_t workspace_bytes() const;
    const Model& model() const;

private:
    Model model_;
    std::vector<float> buffer_a_;
    std::vector<float> buffer_b_;
};

}  // namespace mlp
