#include "mlp/model.hpp"
#include "mlp/quantization.hpp"
#include "mlp/runtime.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/tiny_mlp.txt";

    try {
        mlp::Model model = mlp::load_model(model_path);
        mlp::FloatRuntime float_runtime(model);
        mlp::QuantizedRuntime quantized_runtime(model);

        std::vector<float> input{0.6f, -0.3f, 0.8f, 0.2f};
        std::vector<float> float_output(static_cast<std::size_t>(model.output_size()));
        std::vector<float> quantized_output(static_cast<std::size_t>(model.output_size()));

        float_runtime.infer(input, float_output);
        quantized_runtime.infer(input, quantized_output);

        std::cout << "model: " << model_path << '\n';
        std::cout << "float parameter bytes: " << model.float_bytes() << '\n';
        std::cout << "int8 parameter bytes: " << quantized_runtime.model().storage_bytes() << '\n';
        std::cout << "float workspace bytes: " << float_runtime.workspace_bytes() << '\n';
        std::cout << "int8 workspace bytes: " << quantized_runtime.workspace_bytes() << '\n';

        std::cout << "float output:";
        for (float value : float_output) {
            std::cout << ' ' << std::fixed << std::setprecision(5) << value;
        }
        std::cout << '\n';

        std::cout << "int8 output:";
        for (float value : quantized_output) {
            std::cout << ' ' << std::fixed << std::setprecision(5) << value;
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
