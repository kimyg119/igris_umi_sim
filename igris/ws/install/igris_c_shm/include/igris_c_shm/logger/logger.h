/**
 * @file logger.h
 * @author hs.oh (hs.oh@robros.co.kr)
 * @brief Runtime-named logger with rotating file output, built on spdlog.
 *
 * Obtain a named logger with get_logger("package_name"), analogous to Python's
 * logging.getLogger(). A LoggerRegistry singleton maps names to Logger instances;
 * logging is lock-free once the logger exists.
 *
 * @version 0.1
 * @date 2025-12-30
 * @copyright Copyright (c) 2025
 */

#ifndef __IGRIS_C_SHM_INCLUDE_IGRIS_C_SHM_LOGGER_LOGGER_H__
#define __IGRIS_C_SHM_INCLUDE_IGRIS_C_SHM_LOGGER_LOGGER_H__

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
// spdlog
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
// robros
#include "igris_c_shm/logger/location.h"

namespace robros::shm::logger {

class Logger {
  public:
    explicit Logger(std::filesystem::path path);

    void reopen();
    void set_level(spdlog::level::level_enum level);
    void add_sink(spdlog::sink_ptr sink);

    void log(spdlog::level::level_enum level, std::string_view msg, const source_location &loc = source_location::current());

    void trace(std::string_view m, const source_location &l = source_location::current());
    void debug(std::string_view m, const source_location &l = source_location::current());
    void info(std::string_view m, const source_location &l = source_location::current());
    void warn(std::string_view m, const source_location &l = source_location::current());
    void error(std::string_view m, const source_location &l = source_location::current());
    void fatal(std::string_view m, const source_location &l = source_location::current());

    std::uintmax_t directory_size(const std::filesystem::path &dir_path);
    void cleanup_old_logs(const std::filesystem::path &log_dir, std::uintmax_t max_size);

  private:
    static constexpr std::uintmax_t MAX_FILE_SIZE = 3ULL * 1024 * 1024;         // 3 MB per file
    static constexpr std::size_t MAX_FILES        = 100;                        // 100 files
    static constexpr std::uintmax_t MAX_DIR_SIZE  = 3ULL * 1024 * 1024 * 1024;  // 3 GB

    static std::filesystem::path make_log_path(const std::filesystem::path &log_dir, std::string_view project_name);
    static std::string current_date_str();
    bool is_date_changed();

    void init(const std::filesystem::path &path);

    static std::string format_location(const source_location &loc) {
        std::ostringstream oss;
#ifdef IGRIS_C_LIB_FULL_SOURCE_LOCATION
        oss << loc.file_name() << ":" << loc.line() << " (" << loc.function_name() << ")";
#else
        oss << "(" << loc.function_name() << ")";
#endif
        return oss.str();
    }

    std::shared_ptr<spdlog::logger> logger_;
    std::filesystem::path log_dir_;
    std::string project_name_;
    std::string last_date_;
};

// ---------------------------------------------------------------------------
// LoggerRegistry — singleton map of named loggers (get_logger is thread-safe)
// ---------------------------------------------------------------------------
class LoggerRegistry {
  public:
    static LoggerRegistry &instance() {
        static LoggerRegistry inst;
        return inst;
    }

    Logger &get(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = loggers_.find(name);
        if (it == loggers_.end()) {
            auto [inserted, _] = loggers_.emplace(name, std::make_unique<Logger>(std::filesystem::path("/robros/logs") / name));
            return *inserted->second;
        }
        return *it->second;
    }

  private:
    LoggerRegistry() = default;
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<Logger>> loggers_;
};

inline Logger &get_logger(const std::string &name) { return LoggerRegistry::instance().get(name); }

}  // namespace robros::shm::logger

#endif /* __IGRIS_C_SHM_INCLUDE_IGRIS_C_SHM_LOGGER_LOGGER_H__ */
