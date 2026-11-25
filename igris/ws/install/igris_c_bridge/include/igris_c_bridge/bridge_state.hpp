#pragma once

#include <mutex>
#include <string>
#include <unordered_set>

#include "igris_c_lib/common/robotControlState.h"
#include "igris_c_shm/logger/logger.h"
#include "igris_c_shm/shm/state.h"

namespace robros::bridge {

// Bridge-internal state management.
// No inter-process sharing; IPC is handled by shm/bridge_spsc_types.h (SPSC buffers).
struct BridgeState {
    // Valid motion preset IDs (populated from loaded motion data)
    std::unordered_set<std::string> valid_preset_ids;

    // Robot control state + diagnostics (read from controller via the robot-state SHM).
    std::mutex robot_state_mtx;
    robros::lib::common::RobotControlState robot_control_state_;
    robros::shm::state::robotControlStateData_st control_state_extra_{};
    robros::shm::state::robotIoDiagnosticsData_st io_diag_{};

    // Robot environment (REAL/SIM). launch_environment_ is the bridge's launch-time setting
    // (fallback, set once); the active controller is the authority and overrides it via the
    // robot-state SHM. A REAL/SIM disagreement between the two is logged once.
    uint8_t launch_environment_ = static_cast<uint8_t>(robros::shm::state::RobotEnvironment::REAL);
    uint8_t environment_        = static_cast<uint8_t>(robros::shm::state::RobotEnvironment::REAL);
    bool env_mismatch_warned_   = false;

    void set_robot_control_state(robros::lib::common::RobotControlState state) {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        robot_control_state_ = state;
    }

    robros::lib::common::RobotControlState get_robot_control_state() {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        return robot_control_state_;
    }

    void set_robot_state_extra(const robros::shm::state::robotControlStateData_st &rec) {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        control_state_extra_ = rec;
        environment_         = rec.robot_environment;
        if (rec.robot_environment != launch_environment_ && !env_mismatch_warned_) {
            env_mismatch_warned_ = true;
            const auto env_name  = [](uint8_t e) {
                return e == static_cast<uint8_t>(robros::shm::state::RobotEnvironment::SIM) ? "SIM" : "REAL";
            };
            robros::shm::logger::get_logger("igris_c_bridge")
                .error(std::string("RobotEnvironment mismatch: controller reports ") + env_name(rec.robot_environment) +
                       " but bridge launched as " + env_name(launch_environment_));
        }
    }

    robros::shm::state::robotControlStateData_st get_robot_state_extra() {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        return control_state_extra_;
    }

    // Set the bridge's launch-time environment fallback (REAL/SIM as RobotEnvironment value).
    void set_launch_environment(uint8_t env) {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        launch_environment_ = env;
        environment_        = env;
    }

    // Authoritative environment: the controller-reported value once seen, else the launch fallback.
    uint8_t get_environment() {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        return environment_;
    }

    void set_robot_io_diag(const robros::shm::state::robotIoDiagnosticsData_st &rec) {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        io_diag_ = rec;
    }

    robros::shm::state::robotIoDiagnosticsData_st get_robot_io_diag() {
        std::lock_guard<std::mutex> lk(robot_state_mtx);
        return io_diag_;
    }
};

}  // namespace robros::bridge
