CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude
BUILD_DIR := build
RUNTIME_SRC := src/model.cpp src/quantization.cpp src/runtime.cpp
HEADERS := include/mlp/model.hpp include/mlp/quantization.hpp include/mlp/runtime.hpp

.PHONY: all demo bench test clean

all: demo

demo: $(BUILD_DIR)/mlp_demo

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/mlp_demo: $(RUNTIME_SRC) src/demo.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(RUNTIME_SRC) src/demo.cpp -o $@

bench:
	@printf 'Benchmark target will be added with the benchmark executable.\n'

test:
	@printf 'Test target will be added with runtime tests.\n'

clean:
	rm -rf $(BUILD_DIR)
