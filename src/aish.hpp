#ifndef AISH_HPP
#define AISH_HPP

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Helper function (exposed for testing)
bool is_process_running(pid_t pid);

// Constants for job control
const std::string PID_FILE = "/tmp/aish.pid";
const std::string LOG_FILE = "/tmp/aish.log";

// A custom exception type for aish errors
class AishException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using CommandArgs = std::span<const std::string>;

// Command functions now return their output as a string and throw exceptions on error.
std::string command_echo(CommandArgs args);
std::string command_run(CommandArgs args);
std::string command_status(CommandArgs args);
std::string command_stop(CommandArgs args);
std::string command_monitor(CommandArgs args);

// Main logic for dispatching commands
std::string run_aish(CommandArgs args);

#endif // AISH_HPP
