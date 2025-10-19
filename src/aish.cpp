#include "aish.hpp"

#include <functional>
#include <map>
#include <iostream>

void command_echo(const std::vector<std::string>& args, std::ostream& out) {
    if (!args.empty()) {
        out << "echo output: " << args[0] << std::endl;
    }
}

// Command dispatch table
static std::map<std::string, std::function<void(const std::vector<std::string>&, std::ostream&)>> commands = {
    {"echo", command_echo}
};

int run_aish(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.empty()) {
        err << "Usage: aish <command> [args...]" << std::endl;
        return 1;
    }

    std::string command_name = args[0];
    auto it = commands.find(command_name);

    if (it != commands.end()) {
        std::vector<std::string> command_args(args.begin() + 1, args.end());
        it->second(command_args, out);
    } else {
        err << "That command is not supported in the agent shell (aish)" << std::endl;
        return 1;
    }

    return 0;
}
