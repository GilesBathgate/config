#include "aish.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sys/wait.h>
#include <regex>
#include <filesystem>

class JobControlTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure no leftover files from previous runs
        std::filesystem::remove(PID_FILE);
        std::filesystem::remove(LOG_FILE);
    }

    void TearDown() override {
        // Clean up after each test
        std::filesystem::remove(PID_FILE);
        std::filesystem::remove(LOG_FILE);
    }
};

TEST_F(JobControlTest, FullLifecycle) {
    // 1. Run a job
    std::vector<std::string> run_args = {"run", "sleep", "30"};
    std::string output = run_aish(run_args);
    ASSERT_NE(output.find("Started job 'sleep' with PID"), std::string::npos);

    // 2. Check PID file
    std::ifstream pid_file(PID_FILE);
    ASSERT_TRUE(pid_file.is_open());
    pid_t pid;
    std::string cmd;
    pid_file >> pid >> cmd;
    pid_file.close();
    ASSERT_GT(pid, 0);
    ASSERT_EQ(cmd, "sleep");

    // 3. Check status with regex for the time
    std::vector<std::string> status_args = {"status"};
    output = run_aish(status_args);
    std::regex status_regex("'sleep' \\(pid: \\d+\\) has been running for \\d{2}:\\d{2}:\\d{2}");
    ASSERT_TRUE(std::regex_search(output, status_regex));

    // 4. Try to run another job (should throw)
    ASSERT_THROW(run_aish(run_args), AishException);

    // 5. Stop the job
    std::vector<std::string> stop_args = {"stop"};
    output = run_aish(stop_args);
    ASSERT_NE(output.find("stopped gracefully"), std::string::npos);

    // 6. Verify job is stopped and files are cleaned up
    int status;
    waitpid(pid, &status, 0); // Reap the zombie
    ASSERT_FALSE(is_process_running(pid));
    ASSERT_FALSE(std::filesystem::exists(PID_FILE));
    ASSERT_FALSE(std::filesystem::exists(LOG_FILE));
}

TEST_F(JobControlTest, StalePidFileMessage) {
    // 1. Run a short-lived job
    std::vector<std::string> run_args = {"run", "sleep", "1"};
    run_aish(run_args);

    // 2. Wait for it to finish
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 3. Check status and verify the new message
    std::vector<std::string> status_args = {"status"};
    std::string output = run_aish(status_args);
    std::regex stale_regex("The process \\d+ has already finished and ran for \\d{2}:\\d{2}:\\d{2}");
    ASSERT_TRUE(std::regex_search(output, stale_regex));
}
