#ifndef AISH_HPP
#define AISH_HPP

#include <string>
#include <vector>
#include <iostream>

// Function to handle the echo command
void command_echo(const std::vector<std::string>& args, std::ostream& out = std::cout);

// Main logic for dispatching commands
int run_aish(const std::vector<std::string>& args, std::ostream& out = std::cout, std::ostream& err = std::cerr);

#endif // AISH_HPP
