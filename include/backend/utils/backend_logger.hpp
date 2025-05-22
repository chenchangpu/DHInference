#ifndef DHINFERENCE_LOGGER_H
#define DHINFERENCE_LOGGER_H

#include <string>
#include <iostream>
#include <fstream>
#include <memory>

namespace dhinference {
    namespace backend{

// 日志级别
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    LOG_ERROR,  // 改名避免与Windows.h冲突
    FATAL
};

// 日志工具类
class Logger {
public:
    // 获取单例实例
    static Logger& getInstance();
    
    // 设置日志级别
    void setLogLevel(LogLevel level);
    
    // 设置日志输出文件
    void setLogFile(const std::string& filename);
    
    // 日志输出方法
    void log(LogLevel level, const std::string& message);
    
    // 便捷日志方法
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void fatal(const std::string& message);

private:
    // 构造函数私有化（单例模式）
    Logger();
    ~Logger();
    
    // 禁止拷贝和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // 日志级别
    LogLevel level_;
    
    // 日志文件流
    std::shared_ptr<std::ofstream> file_stream_;
    
    // 将日志级别转换为字符串
    std::string levelToString(LogLevel level);
};

// 宏定义，简化调用
#define LOG_DEBUG(msg) dhinference::backend::Logger::getInstance().debug(msg)
#define LOG_INFO(msg) dhinference::backend::Logger::getInstance().info(msg)
#define LOG_WARNING(msg) dhinference::backend::Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) dhinference::backend::Logger::getInstance().error(msg)
#define LOG_FATAL(msg) dhinference::backend::Logger::getInstance().fatal(msg)

    } // namespace backend
} // namespace dhinference

#endif // DHINFERENCE_LOGGER_H 