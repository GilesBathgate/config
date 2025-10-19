#ifndef AISH_HPP
#define AISH_HPP

#include <string>
#include <vector>
#include <iostream>

// Constants for job control
const std::string PID_FILE = "/tmp/aish.pid";
const std::string LOG_FILE = "/tmp/aish.log";

// Function to handle the echo command
void command_echo(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

// Job control commands
void command_run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);
void command_status(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);
void command_stop(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);
void command_monitor(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);


// Main logic for dispatching commands
int run_aish(const std::vector<std::string>& args, std::ostream& out = std::cout, std::ostream& err = std::cerr);

#endif // AISH_HPP
