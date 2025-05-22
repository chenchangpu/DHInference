#include "backend/utils/backend_logger.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace dhinference {
    namespace backend{

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : level_(LogLevel::INFO), file_stream_(nullptr) {
#ifdef _WIN32
    // 设置Windows控制台为UTF-8编码
    SetConsoleOutputCP(CP_UTF8);
#endif
}

Logger::~Logger() {
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->close();
    }
}

void Logger::setLogLevel(LogLevel level) {
    level_ = level;
}

void Logger::setLogFile(const std::string& filename) {
    file_stream_ = std::make_shared<std::ofstream>(filename, std::ios::out | std::ios::app);
    if (!file_stream_->is_open()) {
        std::cerr << "无法打开日志文件: " << filename << std::endl;
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < level_) {
        return;
    }
    
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << "] "
       << "[" << levelToString(level) << "] "
       << message;
    
    // 输出到控制台
    std::cout << ss.str() << std::endl;
    
    // 输出到文件
    if (file_stream_ && file_stream_->is_open()) {
        *file_stream_ << ss.str() << std::endl;
    }
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::LOG_ERROR, message);
}

void Logger::fatal(const std::string& message) {
    log(LogLevel::FATAL, message);
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::LOG_ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNOWN";
    }
}
    }   // namespace backend
} // namespace dhinference 