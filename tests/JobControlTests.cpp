#include "aish.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sys/wait.h>

class JobControlTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure no leftover files from previous runs
        remove(PID_FILE.c_str());
        remove(LOG_FILE.c_str());
    }

    void TearDown() override {
        // Clean up after each test
        remove(PID_FILE.c_str());
        remove(LOG_FILE.c_str());
    }
};

TEST_F(JobControlTest, FullLifecycle) {
    std::ostringstream out, err;

    // 1. Run a job
    std::vector<std::string> run_args = {"run", "sleep", "30"};
    run_aish(run_args, out, err);
    ASSERT_TRUE(err.str().empty());
    ASSERT_FALSE(out.str().empty());

    // 2. Check PID file
    std::ifstream pid_file(PID_FILE);
    ASSERT_TRUE(pid_file.is_open());
    pid_t pid;
    std::string cmd;
    pid_file >> pid >> cmd;
    pid_file.close();
    ASSERT_GT(pid, 0);
    ASSERT_EQ(cmd, "sleep");

    // 3. Check status
    out.str(""); err.str("");
    std::vector<std::string> status_args = {"status"};
    run_aish(status_args, out, err);
    ASSERT_TRUE(err.str().empty());
    ASSERT_NE(out.str().find("'sleep' (pid: " + std::to_string(pid) + ") has been running for"), std::string::npos);

    // 4. Try to run another job (should fail)
    out.str(""); err.str("");
    run_aish(run_args, out, err);
    ASSERT_FALSE(err.str().empty());
    ASSERT_EQ(err.str(), "Error: A job is already running (pid: " + std::to_string(pid) + ").\n");


    // 5. Stop the job
    out.str(""); err.str("");
    std::vector<std::string> stop_args = {"stop"};
    run_aish(stop_args, out, err);
    ASSERT_TRUE(err.str().empty()) << "stderr was: " << err.str();

    // 6. Verify job is stopped and files are cleaned up
    int status;
    waitpid(pid, &status, 0); // Reap the zombie process

    ASSERT_EQ(kill(pid, 0), -1);
    ASSERT_EQ(errno, ESRCH);
    std::ifstream pid_file_after(PID_FILE);
    ASSERT_FALSE(pid_file_after.is_open());
    std::ifstream log_file_after(LOG_FILE);
    ASSERT_FALSE(log_file_after.is_open());
}

TEST_F(JobControlTest, StalePidFileMessage) {
    std::ostringstream out, err;

    // 1. Run a short-lived job
    std::vector<std::string> run_args = {"run", "sleep", "1"};
    run_aish(run_args, out, err);
    ASSERT_TRUE(err.str().empty());

    // 2. Wait for it to finish
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 3. Check status and verify the new message
    out.str(""); err.str("");
    std::vector<std::string> status_args = {"status"};
    run_aish(status_args, out, err);
    ASSERT_TRUE(err.str().empty());
    ASSERT_NE(out.str().find("has already finished and ran for"), std::string::npos);
}
