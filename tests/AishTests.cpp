#include "aish.hpp"
#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <string>

TEST(AishEchoCommand, PrintsCorrectOutput) {
    std::vector<std::string> args = {"echo", "hello world"};
    std::ostringstream out;
    std::ostringstream err;
    int result = run_aish(args, out, err);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(out.str(), "echo output: hello world\n");
    ASSERT_EQ(err.str(), "");
}

TEST(AishUnknownCommand, PrintsErrorMessage) {
    std::vector<std::string> args = {"unknown_command", "some_arg"};
    std::ostringstream out;
    std::ostringstream err;
    int result = run_aish(args, out, err);
    ASSERT_EQ(result, 1);
    ASSERT_EQ(err.str(), "That command is not supported in the agent shell (aish)\n");
    ASSERT_EQ(out.str(), "");
}

TEST(AishNoCommand, PrintsUsageMessage) {
    std::vector<std::string> args = {};
    std::ostringstream out;
    std::ostringstream err;
    int result = run_aish(args, out, err);
    ASSERT_EQ(result, 1);
    ASSERT_EQ(err.str(), "Usage: aish <command> [args...]\n");
    ASSERT_EQ(out.str(), "");
}
