#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "igris_c_bridge/bridge_state.hpp"
#include "igris_c_shm/logger/logger.h"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::shm {

// Consumes the controller-owned robot-state SHM (control_state_buffer + io_diag_buffer)
// and publishes the latest values into BridgeState for monitoring. Header-only so the
// two bridge entry points (real/sim) can share it without a new build target.
class ShmRobotStateReader {
  public:
    ShmRobotStateReader(std::shared_ptr<BridgeState> state, robros::shm::shm::robotStateShm_st *robot_state_shm)
        : state_(std::move(state)), robot_state_shm_(robot_state_shm) {}

    ~ShmRobotStateReader() { stop(); }

    bool start() {
        if (!robot_state_shm_) {
            robros::shm::logger::get_logger("igris_c_bridge").warn("Robot State SHM not available; controller state monitoring disabled");
            return false;
        }
        running_.store(true);
        feeder_thread_ = std::thread(&ShmRobotStateReader::feederThread, this);
        return true;
    }

    void stop() {
        running_.store(false);
        if (feeder_thread_.joinable()) {
            feeder_thread_.join();
        }
    }

  private:
    void feederThread() {
        std::size_t control_id = 0;
        std::size_t diag_id    = 0;
        if (!robot_state_shm_->control_state_buffer.register_consumer(kConsumerName, control_id, true)) {
            robros::shm::logger::get_logger("igris_c_bridge").error("Failed to register robot control-state consumer");
            return;
        }
        if (!robot_state_shm_->io_diag_buffer.register_consumer(kConsumerName, diag_id, true)) {
            robros::shm::logger::get_logger("igris_c_bridge").error("Failed to register robot io-diagnostics consumer");
            robot_state_shm_->control_state_buffer.unregister_consumer(kConsumerName);
            return;
        }

        robros::shm::state::robotControlStateData_st control_rec;
        robros::shm::state::robotIoDiagnosticsData_st diag_rec;
        bool dropped = false;

        while (running_.load()) {
            if (robot_state_shm_->magic.load(std::memory_order_acquire) != robros::controller::ROBOT_STATE_SHM_MAGIC) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (robot_state_shm_->control_state_buffer.pull_latest(control_id, control_rec, dropped)) {
                state_->set_robot_control_state(robros::lib::common::RobotControlState(
                    static_cast<robros::lib::common::RobotControlState::Value>(control_rec.control_state)));
                state_->set_robot_state_extra(control_rec);
            }
            if (robot_state_shm_->io_diag_buffer.pull_latest(diag_id, diag_rec, dropped)) {
                state_->set_robot_io_diag(diag_rec);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        robot_state_shm_->control_state_buffer.unregister_consumer(kConsumerName);
        robot_state_shm_->io_diag_buffer.unregister_consumer(kConsumerName);
    }

    static constexpr const char *kConsumerName = "bridge";

    std::shared_ptr<BridgeState> state_;
    robros::shm::shm::robotStateShm_st *robot_state_shm_;
    std::thread feeder_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace robros::bridge::shm
