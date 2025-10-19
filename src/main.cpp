#include "aish.hpp"
#include <iostream>
#include <vector>
#include <span>

int main(int argc, char* argv[]) {
    // std::span offers a safe, modern alternative to raw pointers
    std::span<char*> args_span(argv, argc);

    // Convert C-style strings to std::string for application logic
    std::vector<std::string> args;
    for (char* arg : args_span.subspan(1)) { // Skip executable name
        args.emplace_back(arg);
    }

    try {
        std::cout << run_aish(args);
    } catch (const AishException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
