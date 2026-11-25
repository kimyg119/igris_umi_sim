#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <vector>

#include "igris_c_bridge/types.hpp"
#include "igris_c_shm/shm/state.h"

namespace robros::bridge {

// Identifies which command path produced into a single-producer SPSC buffer.
enum class ProducerSource : uint32_t {
    kDds  = 0x1,  // CycloneDDS path (DdsManager -> ShmSensorReader forwarding)
    kRos2 = 0x2,  // rclcpp path (Ros2Manager callbacks)
};

inline robros::shm::state::flags makeDefaultImuFlags() {
    robros::shm::state::flags f{};
    f.dataValid       = 1;
    f.accValid        = 1;
    f.gyroValid       = 1;
    f.rpyValid        = 1;
    f.quatValid       = 1;
    f.serialConnected = 1;
    f.frameSync       = 1;
    f.dataFresh       = 1;
    return f;
}

// Shared data buffer between TCP queries, DDS publishing, and SHM reader.
class DataStore {
  public:
    DataStore();

    // Update methods (called by shm_sensor_reader)
    void updateJointData(const sensor_msgs::msg::JointState &joint_state);
    void updateMotorData(const sensor_msgs::msg::JointState &joint_state, const std::vector<float> &temperatures);
    void updateImuData(const sensor_msgs::msg::Imu &imu_state, const std::array<float, 3> &rpy,
                       const robros::shm::state::flags &status_flags = makeDefaultImuFlags());
    void updateHandData(const robros::shm::state::handV1StateData_st &hand_state);
    void updateRobotStatus(const BridgeRobotStatus &robot_status);

    // Get methods (called by TCP handlers for JSON responses)
    json getJointData();
    json getMotorData();
    json getHandData();
    json getImuData();
    robros::shm::state::flags getImuStatusFlags() const;
    double getBatteryPercent();
    double getUptime() const;
    bool isPowerOn() const;
    bool isInitialized() const;
    bool isTorqueOn() const;
    bool isEmergencyStop() const;
    BridgeRobotStatus getRobotStatus() const;

    // Simulation mode
    void setSimulationMode(bool simulation_mode);
    bool isSimulationMode() const;

    // SPSC single-producer guard. Each command path records its source before
    // pushing into walk_buffer / lowcmd_buffer; if more than one distinct
    // source ever produces, the SPSC single-producer assumption is violated and
    // a warning is logged exactly once per buffer.
    void noteWalkProducer(ProducerSource source);
    void noteLowcmdProducer(ProducerSource source);

    // Notification for data-driven publishing.
    // Returns true if a real sensor update arrived; false on timeout/shutdown.
    // timeout = 0 means wait indefinitely (default).
    bool waitForUpdate(std::atomic<bool> &running, std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
    uint64_t updateSequence() const;
    bool waitForUpdateSince(uint64_t &last_seen_sequence, std::atomic<bool> &running,
                            std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
    void notifyUpdate();

    // Snapshot structs for lock-free DDS publishing
    struct JointSnapshot {
        sensor_msgs::msg::JointState state;
        bool available = false;
    };

    struct MotorSnapshot {
        sensor_msgs::msg::JointState state;
        std::vector<float> temperatures;
        bool available = false;
    };

    struct ImuSnapshot {
        sensor_msgs::msg::Imu state;
        std::array<float, 3> rpy{};
        bool available = false;
    };

    // Snapshot methods (copy data under lock, release, then caller can use freely)
    JointSnapshot snapshotJoint();
    MotorSnapshot snapshotMotor();
    BridgeRobotStatus snapshotRobotStatus(bool &available);
    ImuSnapshot snapshotImu();

  private:
    // Joint data
    std::mutex joint_mtx_;
    sensor_msgs::msg::JointState current_joint_state_;
    bool joint_data_available_{false};

    // Motor data
    std::mutex motor_mtx_;
    sensor_msgs::msg::JointState current_motor_state_;
    std::vector<float> motor_temperatures_;
    bool motor_data_available_{false};

    // Robot status
    mutable std::mutex robot_status_mtx_;
    BridgeRobotStatus current_robot_status_;
    bool robot_status_available_{false};

    // IMU data
    mutable std::mutex imu_mtx_;
    sensor_msgs::msg::Imu current_imu_state_;
    std::array<float, 3> current_imu_rpy_{};
    robros::shm::state::flags current_imu_status_flags_ = makeDefaultImuFlags();
    bool imu_data_available_{false};

    // Hand V1 data
    std::mutex hand_mtx_;
    robros::shm::state::handV1StateData_st current_hand_state_{};
    bool hand_data_available_{false};

    // Uptime
    std::chrono::steady_clock::time_point start_time_;

    // Simulation mode
    std::atomic<bool> simulation_mode_{false};

    // SPSC multi-producer detection (bitmask of ProducerSource values + warn-once latch)
    std::atomic<uint32_t> walk_producers_{0};
    std::atomic<bool> walk_multi_warned_{false};
    std::atomic<uint32_t> lowcmd_producers_{0};
    std::atomic<bool> lowcmd_multi_warned_{false};

    // Data update notification
    mutable std::mutex update_mtx_;
    std::condition_variable update_cv_;
    bool update_pending_{false};
    uint64_t update_sequence_{0};
};

}  // namespace robros::bridge
