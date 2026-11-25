/**
 * @file logger.h
 * @author colson (colson@robros.co.kr)
 * @brief Logger utility for structured logging
 * @version 0.1
 * @date 2025-09-08
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __SRC_IGRIS_C_CAN_INCLUDE_UTIL_LOGGER_H__
#define __SRC_IGRIS_C_CAN_INCLUDE_UTIL_LOGGER_H__

#include <experimental/source_location>
#include <spdlog/common.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace robros {
namespace util {
class Logger {
  public:
    static inline void log(spdlog::logger &lg, spdlog::level::level_enum lvl, const std::string &msg,
                           std::experimental::source_location loc = std::experimental::source_location::current()) {
        lg.log(spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()}, lvl, msg);
    }

    static std::shared_ptr<spdlog::logger> &get() {
        static Logger instance;
        if (!instance.initialized_) {
            instance.init();
            instance.initialized_ = true;
        }
        return instance.logger_;
    }

    static void setFilename(const std::string &filename) {
        auto &instance = get();
        instance->flush_on(spdlog::level::info);
        instance->sinks().clear();

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#][%!] [%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(filename, 1024 * 1024 * 5, 3);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%l] %v");

        instance->sinks().push_back(console_sink);
        instance->sinks().push_back(file_sink);
    }

    void init(const std::string &filename = "log/igris_c_can.log") {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#][%!] [%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(filename, 1024 * 1024 * 5, 3);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%l] %v");

        logger_ = std::make_shared<spdlog::logger>("root", spdlog::sinks_init_list{console_sink, file_sink});
        logger_->set_level(spdlog::level::debug);
        logger_->flush_on(spdlog::level::info);
    }

  private:
    Logger()  = default;
    ~Logger() = default;
    std::shared_ptr<spdlog::logger> logger_;
    bool initialized_ = false;
};
}  // namespace util
}  // namespace robros

#endif
