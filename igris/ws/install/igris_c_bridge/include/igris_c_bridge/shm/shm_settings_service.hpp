#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "igris_c_bridge/types.hpp"
#include "igris_c_shm/logger/logger.h"
#include "igris_c_shm/service/service.h"
#include "igris_c_shm/shm/service.h"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::shm {

// ISettingsService implementation using SHM ServiceClient to communicate with CAN node.
// Connects to 3 separate CAN services: BMS power, torque, and motor initialization.
class ShmSettingsService : public ISettingsService {
  public:
    using ServiceClientType =
        robros::shm::service::ServiceClient<robros::shm::service::OnOffRequest_st, robros::shm::service::OnOffResponse_st>;

    ShmSettingsService() = default;

    // Attach the BMS/torque/init clients to the services embedded in canShm. Call after canShm is
    // open. Returns true once all three service magics are published (else the caller retries).
    bool attachCanServices(robros::shm::shm::canShm_st *can_shm) {
        if (can_shm == nullptr) {
            return false;
        }
        bms_client_    = std::make_shared<ServiceClientType>();
        torque_client_ = std::make_shared<ServiceClientType>();
        init_client_   = std::make_shared<ServiceClientType>();
        const bool ok  = bms_client_->attach(&can_shm->bms_service) && torque_client_->attach(&can_shm->torque_service) &&
                        init_client_->attach(&can_shm->initialize_service);
        if (!ok) {
            bms_client_.reset();
            torque_client_.reset();
            init_client_.reset();
            return false;
        }
        robros::shm::logger::get_logger("igris_c_bridge").info("All CAN ServiceClients attached to canShm");
        return true;
    }

    // Attach the hand-init client to hand_init_service in the hand segment (owned by the hand node,
    // opened later than canShm). Returns true once the hand service magic is published.
    bool attachHandService(robros::shm::shm::handV1Shm_st *hand_shm) {
        if (hand_shm == nullptr) {
            return false;
        }
        hand_init_client_ = std::make_shared<ServiceClientType>();
        return hand_init_client_->attach(&hand_shm->hand_init_service);
    }

    void setTorqueAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) override {
        requestAsync(torque_client_, enable, std::move(cb));
    }

    void setMotorAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) override {
        if (!enable) {
            requestAsync(bms_client_, false, std::move(cb));
            return;
        }

        // Motor ON: BMS_POWER ON -> 2s wait -> MOTOR_INIT ON, all in a single thread
        auto bms_client  = bms_client_;
        auto init_client = init_client_;
        // Guard the startup window before attachCanServices() runs (clients still null). Mirror
        // requestAsync()'s null check so an early command returns an error instead of dereferencing
        // a null ServiceClient and crashing the bridge.
        if (!bms_client || !init_client) {
            cb(false, "ServiceClient not initialized");
            return;
        }
        auto motor_busy = motor_busy_;
        bool expected   = false;
        if (!motor_busy->compare_exchange_strong(expected, true)) {
            cb(false, "Service busy (a motor on is already in progress)");
            return;
        }
        std::thread([bms_client, init_client, cb, motor_busy]() {
            // Clear the busy flag on every exit path of this sequence (RAII).
            struct ClearOnExit {
                std::shared_ptr<std::atomic<bool>> flag;
                ~ClearOnExit() { flag->store(false); }
            } clear_on_exit{motor_busy};

            // Step 1: BMS power on
            const auto power = requestSync(bms_client, true);
            if (!power.first) {
                cb(false, "Power on failed: " + power.second);
                return;
            }

            // Step 2: wait for hardware stabilization
            std::this_thread::sleep_for(std::chrono::seconds(2));

            // Step 3: motor init on
            const auto motor = requestSync(init_client, true);
            cb(motor.first, motor.second);
        }).detach();
    }

    void setBmsPowerAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) override {
        requestAsync(bms_client_, enable, std::move(cb));
    }

    void setMotorInitAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) override {
        requestAsync(init_client_, enable, std::move(cb));
    }

    void setHandInitAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) override {
        requestAsync(hand_init_client_, enable, std::move(cb));
    }

  private:
    static constexpr auto kTimeout = std::chrono::seconds(15);

    // Synchronous core shared by every settings request: null-guard, reject if a request to this
    // service is already in flight (isBusy) — an accidental double-press the CAN worker would
    // otherwise run twice — then issue the blocking request and map the outcome to (ok, message).
    // Static so the detached worker threads never depend on the ShmSettingsService lifetime.
    static std::pair<bool, std::string> requestSync(const std::shared_ptr<ServiceClientType> &client, bool enable) {
        if (!client) {
            return {false, "ServiceClient not initialized"};
        }
        if (client->isBusy()) {
            return {false, "Service busy (a request is already in progress)"};
        }
        robros::shm::service::OnOffRequest_st req{};
        req.onOff = enable;
        robros::shm::service::OnOffResponse_st resp{};
        if (!client->request(req, resp, std::chrono::duration_cast<std::chrono::milliseconds>(kTimeout))) {
            return {false, "Service request timed out"};
        }
        return {resp.success, std::string(resp.message)};
    }

    void requestAsync(std::shared_ptr<ServiceClientType> client, bool enable, std::function<void(bool ok, const std::string &msg)> cb) {
        // Capture the client shared_ptr (not this) so the worker can outlive ShmSettingsService.
        std::thread([client, enable, cb]() {
            const auto r = requestSync(client, enable);
            cb(r.first, r.second);
        }).detach();
    }

    std::shared_ptr<ServiceClientType> bms_client_;
    std::shared_ptr<ServiceClientType> torque_client_;
    std::shared_ptr<ServiceClientType> init_client_;
    std::shared_ptr<ServiceClientType> hand_init_client_;

    // Spans the entire Motor ON sequence (BMS on -> wait -> init on) to reject overlapping Motor ON
    // requests; see setMotorAsync. shared_ptr so a detached worker can outlive `this`.
    std::shared_ptr<std::atomic<bool>> motor_busy_ = std::make_shared<std::atomic<bool>>(false);
};

}  // namespace robros::bridge::shm
