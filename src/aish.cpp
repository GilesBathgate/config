#include "aish.hpp"

#include <chrono>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

// Helper to format duration into HH:MM:SS
std::string format_duration(std::chrono::seconds duration_s) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration_s);
    duration_s -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration_s);
    duration_s -= minutes;

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << hours.count() << ":"
       << std::setfill('0') << std::setw(2) << minutes.count() << ":"
       << std::setfill('0') << std::setw(2) << duration_s.count();
    return ss.str();
}

// Helper to check if a process is running (and not a zombie)
bool is_process_running(pid_t pid) {
    if (kill(pid, 0) != 0) return false;
    std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";
    std::ifstream stat_file(stat_path);
    if (!stat_file) return false;
    std::string state;
    stat_file.ignore(std::numeric_limits<std::streamsize>::max(), ' '); // pid
    stat_file.ignore(std::numeric_limits<std::streamsize>::max(), ')'); // (comm)
    stat_file >> state;
    return state != "Z";
}

long get_job_duration_seconds() {
    try {
        auto start_time = std::filesystem::last_write_time(PID_FILE);
        auto end_time = std::filesystem::last_write_time(LOG_FILE);
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        return duration.count();
    } catch (const std::filesystem::filesystem_error&) {
        return -1;
    }
}


std::string command_echo(CommandArgs args) {
    if (args.empty()) {
        throw AishException("Usage: aish echo <string>");
    }
    return "echo output: " + std::string(args[0]);
}

std::string command_run(CommandArgs args) {
    if (args.empty()) {
        throw AishException("Usage: aish run <command> [args...]");
    }

    if (std::filesystem::exists(PID_FILE)) {
        std::ifstream pid_file(PID_FILE);
        pid_t pid;
        pid_file >> pid;
        if (is_process_running(pid)) {
            throw AishException("Error: A job is already running (pid: " + std::to_string(pid) + ").");
        }
    }

    pid_t pid = fork();
    if (pid < 0) {
        throw AishException("Failed to fork process: " + std::string(strerror(errno)));
    }

    if (pid > 0) { // Parent process
        std::ofstream pid_file(PID_FILE);
        pid_file << pid << " " << args[0];
        return "Started job '" + std::string(args[0]) + "' with PID " + std::to_string(pid);
    } else { // Child process
        if (setsid() < 0) { exit(EXIT_FAILURE); }

        int log_fd = open(LOG_FILE.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (log_fd < 0) { exit(EXIT_FAILURE); }
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        close(log_fd);

        std::vector<char*> exec_args;
        for (const auto& arg : args) {
            exec_args.push_back(const_cast<char*>(arg.c_str()));
        }
        exec_args.push_back(nullptr);

        execvp(exec_args[0], exec_args.data());

        // execvp only returns on error
        fprintf(stderr, "Failed to execute command: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

std::string command_status(CommandArgs args) {
    if (!std::filesystem::exists(PID_FILE)) {
        return "No job is running.";
    }

    std::ifstream pid_file(PID_FILE);
    pid_t pid;
    std::string cmd;
    pid_file >> pid >> cmd;

    if (!is_process_running(pid)) {
        long duration = get_job_duration_seconds();
        std::string msg;
        if (duration < 0) {
            msg = "The process " + std::to_string(pid) + " has already finished.";
        } else {
            msg = "The process " + std::to_string(pid) + " has already finished and ran for " + format_duration(std::chrono::seconds(duration));
        }
        std::filesystem::remove(PID_FILE);
        std::filesystem::remove(LOG_FILE);
        return msg;
    }

    auto start_time = std::filesystem::last_write_time(PID_FILE);
    auto now = std::chrono::file_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

    return "'" + cmd + "' (pid: " + std::to_string(pid) + ") has been running for " + format_duration(duration);
}

std::string command_stop(CommandArgs args) {
     if (!std::filesystem::exists(PID_FILE)) {
        return "No job is running.";
    }

    std::ifstream pid_file(PID_FILE);
    pid_t pid;
    std::string cmd;
    pid_file >> pid >> cmd;

    if (!is_process_running(pid)) {
         long duration = get_job_duration_seconds();
        std::string msg;
        if (duration < 0) {
            msg = "The process " + std::to_string(pid) + " has already finished.";
        } else {
            msg = "The process " + std::to_string(pid) + " has already finished and ran for " + format_duration(std::chrono::seconds(duration));
        }
        std::filesystem::remove(PID_FILE);
        std::filesystem::remove(LOG_FILE);
        return msg;
    }

    std::stringstream output;
    output << "Stopping job '" << cmd << "' (pid: " << pid << ")..." << std::endl;
    if (kill(pid, SIGTERM) != 0) {
        throw AishException("Failed to send SIGTERM to process " + std::to_string(pid) + ": " + strerror(errno));
    }

    for (int i = 0; i < 50; ++i) { // 5s timeout
        if (!is_process_running(pid)) {
            output << "Job '" << cmd << "' stopped gracefully.";
            std::filesystem::remove(PID_FILE);
            std::filesystem::remove(LOG_FILE);
            return output.str();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    output << "Job '" << cmd << "' did not stop within 5 seconds. Sending SIGKILL." << std::endl;
    if (kill(pid, SIGKILL) != 0) {
        throw AishException("Failed to send SIGKILL to process " + std::to_string(pid) + ": " + strerror(errno));
    }

    // Final check
    if(is_process_running(pid)) {
         throw AishException("Process " + std::to_string(pid) + " did not terminate after SIGKILL.");
    }

    output << "Job '" << cmd << "' killed.";
    std::filesystem::remove(PID_FILE);
    std::filesystem::remove(LOG_FILE);
    return output.str();
}

std::string command_monitor(CommandArgs args) {
    if (!std::filesystem::exists(LOG_FILE)) {
        throw AishException("Log file not found. Is a job running?");
    }

    std::stringstream output;
    output << "Monitoring log file: " << LOG_FILE << " for 5 minutes..." << std::endl;

    std::ifstream log_file(LOG_FILE);
    log_file.seekg(0, std::ios::end);

    auto start_time = std::chrono::steady_clock::now();
    while (true) {
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() >= 300) {
            output << "Monitoring timed out after 5 minutes.";
            break;
        }
        std::string line;
        while (std::getline(log_file, line)) {
            output << line << std::endl;
        }
        if (log_file.eof()) {
            log_file.clear();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return output.str();
}

using CommandFunc = std::function<std::string(CommandArgs)>;

static const std::map<std::string, CommandFunc> commands = {
    {"echo", command_echo},
    {"run", command_run},
    {"status", command_status},
    {"stop", command_stop},
    {"monitor", command_monitor}
};

std::string run_aish(CommandArgs args) {
    if (args.empty()) {
        throw AishException("Usage: aish <command> [args...]");
    }

    const auto& command_name = args[0];
    auto it = commands.find(command_name);

    if (it != commands.end()) {
        return it->second(args.subspan(1));
    } else {
        throw AishException("That command is not supported in the agent shell (aish)");
    }
}
