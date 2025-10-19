#include "aish.hpp"

#include <functional>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <cerrno>

// Helper function to calculate the duration of a finished job
long get_job_duration() {
    struct stat pid_stat, log_stat;

    if (stat(PID_FILE.c_str(), &pid_stat) != 0) {
        return -1; // PID file not found
    }

    if (stat(LOG_FILE.c_str(), &log_stat) != 0) {
        return -1; // Log file not found
    }

    auto start_time = std::chrono::system_clock::from_time_t(pid_stat.st_mtime);
    auto end_time = std::chrono::system_clock::from_time_t(log_stat.st_mtime);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    return duration.count();
}

// Helper function to check if a process is running (and not a zombie)
bool is_process_running(pid_t pid) {
    if (kill(pid, 0) != 0) {
        return false; // Process does not exist
    }

    std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";
    std::ifstream stat_file(stat_path);
    if (!stat_file) {
        return false; // Cannot open stat file, assume not running
    }

    // Format is "pid (comm) state ..."
    std::string pid_str, comm, state;
    stat_file >> pid_str >> comm >> state;

    return state != "Z";
}


void command_echo(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (!args.empty()) {
        out << "echo output: " << args[0] << std::endl;
    }
}

// Job control command implementations
void command_run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.empty()) {
        err << "Usage: aish run <command> [args...]" << std::endl;
        return;
    }

    std::ifstream pid_file_check(PID_FILE);
    if (pid_file_check) {
        pid_t pid;
        std::string cmd;
        pid_file_check >> pid >> cmd;
        pid_file_check.close();

        if (is_process_running(pid)) {
            err << "Error: A job is already running (pid: " << pid << ")." << std::endl;
            return;
        }
    }

    pid_t pid = fork();

    if (pid < 0) {
        err << "Failed to fork process: " << strerror(errno) << std::endl;
        return;
    }

    if (pid > 0) {
        std::ofstream pid_file(PID_FILE);
        if (!pid_file) {
            err << "Failed to open PID file for writing: " << PID_FILE << std::endl;
            kill(pid, SIGKILL);
            return;
        }
        pid_file << pid << " " << args[0] << std::endl;
        out << "Started job '" << args[0] << "' with PID " << pid << std::endl;
    } else {
        if (setsid() < 0) {
            exit(EXIT_FAILURE);
        }

        int log_fd = open(LOG_FILE.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (log_fd < 0) {
            exit(EXIT_FAILURE);
        }
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        close(log_fd);

        std::vector<char*> exec_args;
        for (const auto& arg : args) {
            exec_args.push_back(const_cast<char*>(arg.c_str()));
        }
        exec_args.push_back(nullptr);

        execvp(exec_args[0], exec_args.data());

        fprintf(stderr, "Failed to execute command: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

#include <sys/stat.h>
#include <chrono>

void command_status(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    std::ifstream pid_file(PID_FILE);
    if (!pid_file) {
        out << "No job is running." << std::endl;
        return;
    }

    pid_t pid;
    std::string cmd;
    pid_file >> pid >> cmd;
    pid_file.close();

    if (!is_process_running(pid)) {
        long duration = get_job_duration();
        if (duration < 0) {
            out << "The process " << pid << " has already finished." << std::endl;
        } else if (duration > 86400) { // 24 hours
            out << "The process " << pid << " has already finished and ran for a very long time" << std::endl;
        } else {
            out << "The process " << pid << " has already finished and ran for " << duration << "s" << std::endl;
        }
        remove(PID_FILE.c_str());
        remove(LOG_FILE.c_str());
        return;
    }

    struct stat pid_stat;
    if (stat(PID_FILE.c_str(), &pid_stat) != 0) {
        err << "Failed to get status of PID file: " << strerror(errno) << std::endl;
        return;
    }

    auto start_time = std::chrono::system_clock::from_time_t(pid_stat.st_mtime);
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

    long total_seconds = duration.count();
    long hours = total_seconds / 3600;
    long minutes = (total_seconds % 3600) / 60;
    long seconds = total_seconds % 60;

    out << "'" << cmd << "' (pid: " << pid << ") has been running for ";
    if (hours > 0) out << hours << "h ";
    if (minutes > 0) out << minutes << "m ";
    out << seconds << "s" << std::endl;
}

#include <thread>

void command_stop(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    std::ifstream pid_file(PID_FILE);
    if (!pid_file) {
        out << "No job is running." << std::endl;
        return;
    }

    pid_t pid;
    std::string cmd;
    pid_file >> pid >> cmd;
    pid_file.close();

    if (!is_process_running(pid)) {
        long duration = get_job_duration();
        if (duration < 0) {
            out << "The process " << pid << " has already finished." << std::endl;
        } else if (duration > 86400) { // 24 hours
            out << "The process " << pid << " has already finished and ran for a very long time" << std::endl;
        } else {
            out << "The process " << pid << " has already finished and ran for " << duration << "s" << std::endl;
        }
        remove(PID_FILE.c_str());
        remove(LOG_FILE.c_str());
        return;
    }

    out << "Stopping job '" << cmd << "' (pid: " << pid << ")..." << std::endl;
    if (kill(pid, SIGTERM) != 0) {
        err << "Failed to send SIGTERM to process " << pid << ": " << strerror(errno) << std::endl;
        return;
    }

    bool stopped_gracefully = false;
    for (int i = 0; i < 50; ++i) { // 50 * 100ms = 5 seconds
        if (!is_process_running(pid)) {
            stopped_gracefully = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (stopped_gracefully) {
        out << "Job '" << cmd << "' stopped gracefully." << std::endl;
    } else {
        out << "Job '" << cmd << "' did not stop within 5 seconds. Sending SIGKILL." << std::endl;
        if (kill(pid, SIGKILL) != 0) {
            err << "Failed to send SIGKILL to process " << pid << ": " << strerror(errno) << std::endl;
            return; // Don't clean up if kill failed
        }

        // Wait for the process to be killed
        bool killed = false;
        for (int i = 0; i < 50; ++i) { // Wait up to 5 seconds
            if (!is_process_running(pid)) {
                killed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (killed) {
            out << "Job '" << cmd << "' killed." << std::endl;
        } else {
            err << "Process " << pid << " did not terminate after SIGKILL." << std::endl;
            return; // Don't clean up if it's still alive
        }
    }

    remove(PID_FILE.c_str());
    remove(LOG_FILE.c_str());
}

void command_monitor(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    std::ifstream log_file(LOG_FILE);
    if (!log_file) {
        err << "Log file not found. Is a job running?" << std::endl;
        return;
    }

    out << "Monitoring log file: " << LOG_FILE << " for 5 minutes..." << std::endl;
    log_file.seekg(0, std::ios::end);

    auto start_time = std::chrono::steady_clock::now();
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= 300) {
            out << "Monitoring timed out after 5 minutes." << std::endl;
            break;
        }

        std::string line;
        while (std::getline(log_file, line)) {
            out << line << std::endl;
        }

        if (log_file.eof()) {
            log_file.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } else {
            err << "Error reading log file." << std::endl;
            break;
        }
    }
}


// Command dispatch table
static std::map<std::string, std::function<void(const std::vector<std::string>&, std::ostream&, std::ostream&)>> commands = {
    {"echo", command_echo},
    {"run", command_run},
    {"status", command_status},
    {"stop", command_stop},
    {"monitor", command_monitor}
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
        it->second(command_args, out, err);
    } else {
        err << "That command is not supported in the agent shell (aish)" << std::endl;
        return 1;
    }

    return 0;
}
