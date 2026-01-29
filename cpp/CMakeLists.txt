cmake_minimum_required(VERSION 3.18)
project(trade_degradation_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put the built Python extension module into ../python for easy importing
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../python)

include(FetchContent)

# ---- Python (headers + libs) ----
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

# ---- pybind11 ----
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ---- Module ----
pybind11_add_module(trade_degradation_cpp
  bindings.cpp
  degradation.cpp
)

target_include_directories(trade_degradation_cpp PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${Python3_INCLUDE_DIRS}
)

target_link_libraries(trade_degradation_cpp PRIVATE
  Python3::Python
)
