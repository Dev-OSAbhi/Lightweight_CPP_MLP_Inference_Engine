CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude
BUILD_DIR := build

.PHONY: all demo bench test clean

all:
	@printf 'Targets will be added as the runtime is implemented.\n'

demo:
	@printf 'Demo target will be added with the float runtime.\n'

bench:
	@printf 'Benchmark target will be added with the benchmark executable.\n'

test:
	@printf 'Test target will be added with runtime tests.\n'

clean:
	rm -rf $(BUILD_DIR)
