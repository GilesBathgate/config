cmake_minimum_required(VERSION 3.14)
project(aish)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add library for core logic
add_library(aish_lib src/aish.cpp)
target_include_directories(aish_lib PUBLIC src)

# Add executable
add_executable(aish src/main.cpp)
target_link_libraries(aish PRIVATE aish_lib)

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add test executable
add_executable(aish_tests tests/AishTests.cpp)
target_link_libraries(aish_tests PRIVATE GTest::gtest_main aish_lib)

# Add test
include(GoogleTest)
gtest_discover_tests(aish_tests)
