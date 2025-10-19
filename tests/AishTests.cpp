#include "aish.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <string>

TEST(AishEchoCommand, PrintsCorrectOutput) {
    std::vector<std::string> args = {"echo", "hello world"};
    std::string output = run_aish(args);
    ASSERT_EQ(output, "echo output: hello world");
}

TEST(AishUnknownCommand, ThrowsException) {
    std::vector<std::string> args = {"unknown_command", "some_arg"};
    ASSERT_THROW(run_aish(args), AishException);
}

TEST(AishNoCommand, ThrowsException) {
    std::vector<std::string> args = {};
    ASSERT_THROW(run_aish(args), AishException);
}
