/**
 * @file sharedMemory.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-12-30
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_SHM_SHAREDMEMORY_H__
#define __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_SHM_SHAREDMEMORY_H__

#include "igris_c_shm/logger/logger.h"

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace robros::shm::shm {
enum class ShmMode { CREATE, OPEN };

// Controls whether CREATE zero-fills the segment.
//   Zero     - memset the segment to 0 on create (clean slate; the default).
//   Preserve - attach to / re-create WITHOUT wiping existing contents, e.g. to keep the SPMC
//              write head + consumer registration table across a single producer restart.
enum class ZeroOnCreate { Zero, Preserve };

inline std::string normalizeShmName(const char *raw_name) {
    std::string name = raw_name ? std::string(raw_name) : std::string();

    if (name.empty()) {
        return "/shm_default";
    }

    if (name.front() != '/') {
        name.insert(name.begin(), '/');
    }

    for (std::size_t i = 1; i < name.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        if (name[i] == '/' || !(std::isalnum(c) || name[i] == '_' || name[i] == '-' || name[i] == '.')) {
            name[i] = '_';
        }
    }

    if (name.size() == 1) {
        return "/shm_default";
    }

    return name;
}

template <typename T> class SharedMemoryManager {
  public:
    SharedMemoryManager() = delete;
    SharedMemoryManager(const char *shm_name, bool autoCleanup);
    ~SharedMemoryManager();

    bool initialize(ShmMode create = ShmMode::OPEN, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000),
                    ZeroOnCreate zeroOnCreate = ZeroOnCreate::Zero);
    T *get_shm();

  private:
    std::string shm_name_;
    int shm_fd_;
    T *shm_;
    bool is_initialized_;
    bool auto_cleanup_;
};

template <typename T>
SharedMemoryManager<T>::SharedMemoryManager(const char *shm_name, bool autoCleanup)
    : shm_name_(normalizeShmName(shm_name)), shm_fd_(-1), shm_(nullptr), is_initialized_(false), auto_cleanup_(autoCleanup) {
    robros::shm::logger::get_logger("igris_c_shm").info(std::string("SharedMemoryManager created for: ") + shm_name_);
}

template <typename T> SharedMemoryManager<T>::~SharedMemoryManager() {
    if (is_initialized_) {
        if (shm_ != nullptr) {
            munmap(shm_, sizeof(T));
            shm_ = nullptr;
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
            shm_fd_ = -1;
        }
        if (auto_cleanup_) {
            shm_unlink(shm_name_.c_str());
        }
    } else if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}
template <typename T>
bool SharedMemoryManager<T>::initialize(ShmMode create, std::chrono::milliseconds timeout, ZeroOnCreate zeroOnCreate) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        if (create == ShmMode::CREATE)
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR | O_CREAT, 0666);
        else
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);

        if (shm_fd_ != -1)
            break;

        if (std::chrono::steady_clock::now() - start > timeout) {
            robros::shm::logger::get_logger("igris_c_shm")
                .error("Failed to open shared memory: " + shm_name_ + " after " + std::to_string(timeout.count()) + " ms");
            return false;
        }
        robros::shm::logger::get_logger("igris_c_shm").warn("Shared memory not ready, retrying... : " + std::string(strerror(errno)));
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (ftruncate(shm_fd_, sizeof(T)) == -1) {
        robros::shm::logger::get_logger("igris_c_shm").error("Failed to truncate shared memory: " + std::string(strerror(errno)));
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    shm_ = reinterpret_cast<T *>(mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0));
    if (shm_ == MAP_FAILED) {
        robros::shm::logger::get_logger("igris_c_shm").error("Failed to map shared memory: " + std::string(strerror(errno)));
        close(shm_fd_);
        shm_fd_ = -1;
        shm_    = nullptr;
        return false;
    }

    if (create == ShmMode::CREATE && zeroOnCreate == ZeroOnCreate::Zero)
        std::memset(shm_, 0, sizeof(T));

    is_initialized_ = true;
    return true;
}

template <typename T> T *SharedMemoryManager<T>::get_shm() { return shm_; }
}  // namespace robros::shm::shm

#endif /* __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_SHM_SHAREDMEMORY_H__ */
