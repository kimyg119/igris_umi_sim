/**
 * @file service.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2026-03-11
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_SERVICE_SERVICE_H__
#define __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_SERVICE_SERVICE_H__

#include "igris_c_shm/shm/sharedMemory.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace robros::shm::service {

template <typename reqType, typename respType> struct ServiceSharedMemory {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> request_seq;
    std::atomic<uint32_t> response_seq;
    reqType request;
    respType response;
};

template <typename reqType, typename respType> class ServiceServer {
    static_assert(std::is_trivially_copyable_v<reqType>, "reqType must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<respType>, "respType must be trivially copyable");

  public:
    static constexpr uint32_t SERVICE_MAGIC = 0x53455256;  // 'SERV'

    ServiceServer(const char *shm_name, bool auto_cleanup = false,
                  std::chrono::milliseconds spin_poll_interval = std::chrono::milliseconds(1))
        : shm_name_(shm_name), auto_cleanup_(auto_cleanup), shm_ptr_(nullptr), last_processed_seq_(0), handler_(nullptr), running_(false),
          spin_poll_interval_(spin_poll_interval) {}

    // Co-located construction: owns no segment. Use with attach() to serve a ServiceSharedMemory
    // that lives inside another SHM segment.
    explicit ServiceServer(std::chrono::milliseconds spin_poll_interval)
        : shm_name_(nullptr), auto_cleanup_(false), shm_ptr_(nullptr), last_processed_seq_(0), handler_(nullptr), running_(false),
          spin_poll_interval_(spin_poll_interval) {}

    ~ServiceServer() { shutdown(); }

    ServiceServer(const ServiceServer &)            = delete;
    ServiceServer &operator=(const ServiceServer &) = delete;
    ServiceServer(ServiceServer &&)                 = delete;
    ServiceServer &operator=(ServiceServer &&)      = delete;

    bool initialize() {
        shm_manager_ =
            std::make_unique<robros::shm::shm::SharedMemoryManager<ServiceSharedMemory<reqType, respType>>>(shm_name_, auto_cleanup_);
        if (!shm_manager_->initialize(robros::shm::shm::ShmMode::CREATE)) {
            return false;
        }

        shm_ptr_ = shm_manager_->get_shm();
        shm_ptr_->magic.store(SERVICE_MAGIC, std::memory_order_release);
        shm_ptr_->request_seq.store(0, std::memory_order_release);
        shm_ptr_->response_seq.store(0, std::memory_order_release);
        last_processed_seq_ = 0;
        startAutoSpin();

        return true;
    }

    // Attach to a ServiceSharedMemory that lives inside another SHM segment (the caller owns that
    // segment's lifecycle), instead of creating an owned segment. Reuses the same auto-spin worker +
    // handler as initialize(), so co-located services need no hand-rolled polling. Call setHandler()
    // afterwards to start serving.
    // auto_spin=true: a background worker serves requests (thread-safe handlers, e.g. CAN/hand).
    // auto_spin=false: no worker; the owner must call spinOnce() from its own loop (use when the
    // handler must run on a specific thread, e.g. the single-threaded MuJoCo sim loop).
    bool attach(ServiceSharedMemory<reqType, respType> *service, bool auto_spin = true) {
        if (service == nullptr) {
            return false;
        }
        auto_spin_ = auto_spin;
        shm_ptr_   = service;
        shm_ptr_->magic.store(SERVICE_MAGIC, std::memory_order_release);
        shm_ptr_->request_seq.store(0, std::memory_order_release);
        shm_ptr_->response_seq.store(0, std::memory_order_release);
        last_processed_seq_ = 0;
        if (auto_spin_) {
            startAutoSpin();
        }
        return true;
    }

    void shutdown() {
        if (shm_ptr_) {
            shm_ptr_->magic.store(0, std::memory_order_release);
        }
        stopAutoSpin();
    }

    void setHandler(std::function<void(const reqType &, respType &)> handler) {
        {
            std::lock_guard<std::mutex> lock(handler_mutex_);
            handler_ = std::move(handler);
        }
        if (auto_spin_) {
            startAutoSpin();
        }
    }

    bool spinOnce() {
        if (!shm_ptr_) {
            return false;
        }

        std::function<void(const reqType &, respType &)> handler;
        {
            std::lock_guard<std::mutex> lock(handler_mutex_);
            if (!handler_) {
                return false;
            }
            handler = handler_;
        }

        std::lock_guard<std::mutex> lock(spin_mutex_);

        const uint32_t request_seq = shm_ptr_->request_seq.load(std::memory_order_acquire);
        if (request_seq == last_processed_seq_) {
            return false;
        }

        const reqType request = shm_ptr_->request;
        respType response{};
        handler(request, response);

        shm_ptr_->response = response;
        shm_ptr_->response_seq.store(request_seq, std::memory_order_release);
        last_processed_seq_ = request_seq;
        return true;
    }

  private:
    void startAutoSpin() {
        if (running_.load(std::memory_order_acquire) || !shm_ptr_) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(handler_mutex_);
            if (!handler_) {
                return;
            }
        }

        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }

        worker_thread_ = std::thread([this]() {
            while (running_.load(std::memory_order_acquire)) {
                const bool processed = spinOnce();
                if (!processed && spin_poll_interval_ > std::chrono::milliseconds(0)) {
                    std::this_thread::sleep_for(spin_poll_interval_);
                }
            }
        });
    }

    void stopAutoSpin() {
        const bool was_running = running_.exchange(false, std::memory_order_acq_rel);
        if (was_running && worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    const char *shm_name_;
    bool auto_cleanup_;
    std::unique_ptr<robros::shm::shm::SharedMemoryManager<ServiceSharedMemory<reqType, respType>>> shm_manager_;
    ServiceSharedMemory<reqType, respType> *shm_ptr_;
    uint32_t last_processed_seq_;
    std::function<void(const reqType &, respType &)> handler_;
    std::thread worker_thread_;
    std::atomic<bool> running_;
    std::chrono::milliseconds spin_poll_interval_;
    bool auto_spin_ = true;  // false => owner drives spinOnce() (no worker thread)
    std::mutex handler_mutex_;
    std::mutex spin_mutex_;
};

template <typename reqType, typename respType> class ServiceClient {
    static_assert(std::is_trivially_copyable_v<reqType>, "reqType must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<respType>, "respType must be trivially copyable");

  public:
    static constexpr uint32_t SERVICE_MAGIC = 0x53455256;  // 'SERV'

    ServiceClient(const char *shm_name, bool auto_cleanup = false) : shm_name_(shm_name), auto_cleanup_(auto_cleanup), shm_ptr_(nullptr) {}

    // Co-located construction: owns no segment. Use with attach() to talk to a ServiceSharedMemory
    // that lives inside another SHM segment.
    ServiceClient() : shm_name_(nullptr), auto_cleanup_(false), shm_ptr_(nullptr) {}

    bool initialize() {
        shm_manager_ =
            std::make_unique<robros::shm::shm::SharedMemoryManager<ServiceSharedMemory<reqType, respType>>>(shm_name_, auto_cleanup_);
        if (!shm_manager_->initialize(robros::shm::shm::ShmMode::OPEN)) {
            return false;
        }

        shm_ptr_ = shm_manager_->get_shm();
        return isReady();
    }

    // Attach to a ServiceSharedMemory that lives inside another (caller-owned) SHM segment instead
    // of opening a standalone segment. Ready once the server has published its magic.
    bool attach(ServiceSharedMemory<reqType, respType> *service) {
        shm_ptr_ = service;
        return isReady();
    }

    bool isReady() const { return shm_ptr_ && shm_ptr_->magic.load(std::memory_order_acquire) == SERVICE_MAGIC; }

    bool isBusy() const {
        if (!isReady()) {
            return false;
        }

        const uint32_t request_seq  = shm_ptr_->request_seq.load(std::memory_order_acquire);
        const uint32_t response_seq = shm_ptr_->response_seq.load(std::memory_order_acquire);
        return request_seq != response_seq;
    }

    bool request(const reqType &request, respType &response, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000),
                 std::chrono::milliseconds poll_interval = std::chrono::milliseconds(1)) {
        std::lock_guard<std::mutex> request_lock(request_mutex_);
        if (!isReady()) {
            return false;
        }

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (isBusy()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }

            if (poll_interval > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(poll_interval);
            }
        }

        const uint32_t request_seq = shm_ptr_->request_seq.load(std::memory_order_relaxed) + 1;
        shm_ptr_->request          = request;
        shm_ptr_->request_seq.store(request_seq, std::memory_order_release);

        while (true) {
            if (!isReady()) {
                break;
            }

            if (shm_ptr_->response_seq.load(std::memory_order_acquire) == request_seq) {
                response = shm_ptr_->response;
                return true;
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }

            if (poll_interval > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(poll_interval);
            }
        }

        return false;
    }

  private:
    const char *shm_name_;
    bool auto_cleanup_;
    std::unique_ptr<robros::shm::shm::SharedMemoryManager<ServiceSharedMemory<reqType, respType>>> shm_manager_;
    ServiceSharedMemory<reqType, respType> *shm_ptr_;
    mutable std::mutex request_mutex_;
};
}  // namespace robros::shm::service

#endif /* __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_SERVICE_SERVICE_H__ */
