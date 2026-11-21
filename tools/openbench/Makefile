REPO_ROOT := $(abspath ../..)
BUILD_DIR ?= $(REPO_ROOT)/build/openbench-release
EXE ?= localchessengine
CXX ?= clang++
CMAKE ?= cmake
CMAKE_GENERATOR ?= Ninja
VCPKG_ROOT ?= $(REPO_ROOT)/.deps/vcpkg
VCPKG_TOOLCHAIN := $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake

.PHONY: all bench clean

all:
	@if [ ! -f "$(VCPKG_TOOLCHAIN)" ]; then \
		echo "Missing vcpkg toolchain: $(VCPKG_TOOLCHAIN)" >&2; \
		echo "Set VCPKG_ROOT or bootstrap the project-local .deps/vcpkg first." >&2; \
		exit 2; \
	fi
	$(CMAKE) -S "$(REPO_ROOT)" -B "$(BUILD_DIR)" -G "$(CMAKE_GENERATOR)" \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER="$(CXX)" \
		-DCMAKE_TOOLCHAIN_FILE="$(VCPKG_TOOLCHAIN)" \
		-DVCPKG_TARGET_TRIPLET=arm64-osx \
		-DBUILD_TESTING=OFF
	$(CMAKE) --build "$(BUILD_DIR)" --target localchessengine
	cp "$(BUILD_DIR)/engine/localchessengine" "$(CURDIR)/$(EXE)"

bench: all
	"$(CURDIR)/$(EXE)" bench

clean:
	rm -f "$(CURDIR)/$(EXE)"
