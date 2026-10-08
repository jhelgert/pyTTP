cmake_minimum_required(VERSION 3.26...4.0)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT SKBUILD)
  message(WARNING "Build this project with `uv build` / `pip install .`.")
endif()

# Release by default; the wheel build does not set a build type itself.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Tune for the CPU of the build machine. Great for local builds, but the result
# may crash on older CPUs, so release wheels turn this off (see pyproject.toml).
option(PYTTP_NATIVE "Compile for the host CPU (-march=native / /arch:AVX2)" ON)
set(PYTTP_BASELINE "" CACHE STRING "Explicit -march value used when PYTTP_NATIVE is OFF")

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Opt-in stable ABI wheels: set `wheel.py-api = "cp312"` in pyproject.toml (Python >= 3.12).
if(DEFINED SKBUILD_SABI_VERSION AND SKBUILD_SABI_VERSION)
  set(PYTTP_STABLE_ABI STABLE_ABI)
else()
  set(PYTTP_STABLE_ABI "")
endif()

nanobind_add_module(
  _core
  ${PYTTP_STABLE_ABI}
  NB_STATIC
  NOMINSIZE
  src/cpp/bindings.cpp
  src/cpp/evaluate.cpp
  src/cpp/local_search.cpp
  src/cpp/neighborhoods.cpp)

target_include_directories(_core PRIVATE src/cpp)
target_compile_features(_core PRIVATE cxx_std_23)

if(MSVC)
  target_compile_options(_core PRIVATE /O2 /W4 /permissive-)
  if(PYTTP_NATIVE)
    target_compile_options(_core PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
  if(PYTTP_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
  elseif(PYTTP_BASELINE)
    target_compile_options(_core PRIVATE -march=${PYTTP_BASELINE})
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION pyttp)
