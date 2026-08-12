#include "mlp/model.hpp"
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
        mlp::FloatRuntime runtime(std::move(model));

        std::vector<float> input{0.6f, -0.3f, 0.8f, 0.2f};
        std::vector<float> output(static_cast<std::size_t>(runtime.model().output_size()));

        runtime.infer(input, output);

        std::cout << "model: " << model_path << '\n';
        std::cout << "float parameters: " << runtime.model().parameter_count() << '\n';
        std::cout << "workspace bytes: " << runtime.workspace_bytes() << '\n';
        std::cout << "output:";
        for (float value : output) {
            std::cout << ' ' << std::fixed << std::setprecision(5) << value;
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
