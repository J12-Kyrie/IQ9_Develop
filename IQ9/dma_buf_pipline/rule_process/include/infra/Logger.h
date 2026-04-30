#ifndef INFRA_LOGGER_H
#define INFRA_LOGGER_H

#include <iostream>
#include <mutex>
#include <string>

namespace utils {

enum class LogLevel { INFO, WARN, ERROR };

class Logger {
public:
    static void log(LogLevel level, const char* tag, const std::string& message)
    {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);
        static const char* names[] = {"INFO", "WARN", "ERROR"};
        auto& os = (level >= LogLevel::WARN) ? std::cerr : std::cout;
        os << "[" << names[static_cast<int>(level)] << "][" << tag << "] " << message << std::endl;
    }

    static void info(const char* tag, const std::string& msg)  { log(LogLevel::INFO, tag, msg); }
    static void warn(const char* tag, const std::string& msg)  { log(LogLevel::WARN, tag, msg); }
    static void error(const char* tag, const std::string& msg) { log(LogLevel::ERROR, tag, msg); }

    // Backward-compatible overloads (used by MQTT modules)
    static void info(const std::string& msg)  { log(LogLevel::INFO, "?", msg); }
    static void warn(const std::string& msg)  { log(LogLevel::WARN, "?", msg); }
    static void error(const std::string& msg) { log(LogLevel::ERROR, "?", msg); }
};

} // namespace utils

#endif // INFRA_LOGGER_H
