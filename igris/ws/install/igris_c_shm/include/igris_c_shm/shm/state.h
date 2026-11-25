/**
 * @file state.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __STATE_DED39D17_AA65_48AD_844E_F2A6BA78DA93_H__
#define __STATE_DED39D17_AA65_48AD_844E_F2A6BA78DA93_H__

#include "igris_c/common/robot_define.h"
#include "igris_c_shm/shm/common.h"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace robros::shm::state {

struct jointStatusData_st {
    using VectorQd = Eigen::Matrix<double, MODEL_DOF, 1>;

    common::Header header;
    uint32_t joint_id[MODEL_DOF];
    char joint_name[MODEL_DOF][50];
    double current_position[MODEL_DOF];  // radians or degrees
    double current_velocity[MODEL_DOF];  // rad/s or deg/s
    double current_torque[MODEL_DOF];    // Nm
    uint32_t error_state[MODEL_DOF];     // CAN status word

    jointStatusData_st() : header() {
        std::memset(joint_id, 0, sizeof(joint_id));
        std::memset(joint_name, 0, sizeof(joint_name));
        std::memset(current_position, 0, sizeof(current_position));
        std::memset(current_velocity, 0, sizeof(current_velocity));
        std::memset(current_torque, 0, sizeof(current_torque));
        std::memset(error_state, 0, sizeof(error_state));
    }

    auto qVector() const {
        auto q = Eigen::Map<const VectorQd>(current_position, MODEL_DOF);
        return q;
    }

    auto qdVector() const {
        auto qd = Eigen::Map<const VectorQd>(current_velocity, MODEL_DOF);
        return qd;
    }

    auto qddVector() const {
        // Not provided by hardware, return zero vector for now
        auto qdd = Eigen::Matrix<double, MODEL_DOF, 1>::Zero();
        return qdd;
    }

    auto torqueVector() const {
        auto tau = Eigen::Map<const VectorQd>(current_torque, MODEL_DOF);
        return tau;
    }

    auto errorStateVector() const {
        auto err = Eigen::Map<const Eigen::Matrix<uint32_t, MODEL_DOF, 1>>(error_state, MODEL_DOF);
        return err;
    }

    auto q(int idx) const {
        if (idx < MODEL_DOF) {
            return current_position[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    auto qDot(int idx) const {
        if (idx < MODEL_DOF) {
            return current_velocity[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    auto torque(int idx) const {
        if (idx < MODEL_DOF) {
            return current_torque[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    auto error(int idx) const {
        if (idx < MODEL_DOF) {
            return error_state[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    std::string toString() {
        std::stringstream sstr;
        sstr << "Motor State:\n";
        auto format_value = [](double val) { return std::abs(val) < 0.001 ? 0.0 : val; };
        sstr << std::fixed << std::setprecision(3);
        sstr << " Joint Positions: [" << qVector().unaryExpr(format_value).transpose() << "]\n";
        sstr << " Joint Velocities: [" << qdVector().unaryExpr(format_value).transpose() << "]\n";
        sstr << " Joint Torques: [" << torqueVector().unaryExpr(format_value).transpose() << "]\n";
        sstr << " Error State: [" << errorStateVector().transpose() << "]\n";
        return sstr.str();
    }
};

struct motorStatusData_st {
    common::Header header;
    uint32_t motor_id[MODEL_DOF];
    char joint_name[MODEL_DOF][30];
    char motor_name[MODEL_DOF][30];
    double target_position[MODEL_DOF];
    double target_torque[MODEL_DOF];
    double current_position[MODEL_DOF];      // rad
    double current_velocity[MODEL_DOF];      // rad/s or deg/s
    double current_acceleration[MODEL_DOF];  // rad/s^2 or deg/s^2, not provided by hardware, set to zero for now
    double current_torque[MODEL_DOF];        // Nm
    double current_current[MODEL_DOF];       //
    double current_temp[MODEL_DOF];          //
    uint32_t error_state[MODEL_DOF];         // CAN status word
    uint64_t packet_rate[MODEL_DOF];         // CAN frame count

    motorStatusData_st() : header() {
        std::memset(motor_id, 0, sizeof(motor_id));
        std::memset(joint_name, 0, sizeof(joint_name));
        std::memset(motor_name, 0, sizeof(motor_name));
        std::memset(target_position, 0, sizeof(target_position));
        std::memset(target_torque, 0, sizeof(target_torque));
        std::memset(current_position, 0, sizeof(current_position));
        std::memset(current_velocity, 0, sizeof(current_velocity));
        std::memset(current_acceleration, 0, sizeof(current_acceleration));
        std::memset(current_torque, 0, sizeof(current_torque));
        std::memset(current_current, 0, sizeof(current_current));
        std::memset(current_temp, 0, sizeof(current_temp));
        std::memset(error_state, 0, sizeof(error_state));
        std::memset(packet_rate, 0, sizeof(packet_rate));
    }

    void update(int idx, uint32_t motorId, const std::string &jointName, const std::string &motorName, uint32_t errorState, double position,
                double velocity, double torque, double current, double temp, uint64_t rate) {
        motor_id[idx] = motorId;
        std::memcpy(joint_name[idx], jointName.c_str(), std::min(jointName.size(), sizeof(joint_name[idx]) - 1));
        std::memcpy(motor_name[idx], motorName.c_str(), std::min(motorName.size(), sizeof(motor_name[idx]) - 1));
        error_state[idx]      = errorState;
        current_position[idx] = position;
        current_velocity[idx] = velocity;
        current_torque[idx]   = torque;
        current_current[idx]  = current;
        current_temp[idx]     = temp;
        packet_rate[idx]      = rate;
    }

    auto qVector() const {
        auto q = Eigen::Map<const Eigen::Matrix<double, MODEL_DOF, 1>>(current_position, MODEL_DOF);
        return q;
    }

    auto qdVector() const {
        auto qd = Eigen::Map<const Eigen::Matrix<double, MODEL_DOF, 1>>(current_velocity, MODEL_DOF);
        return qd;
    }

    auto qddVector() const {
        // Not provided by hardware, return zero vector for now
        auto qdd = Eigen::Matrix<double, MODEL_DOF, 1>::Zero();
        return qdd;
    }

    auto torqueVector() const {
        auto tau = Eigen::Map<const Eigen::Matrix<double, MODEL_DOF, 1>>(current_torque, MODEL_DOF);
        return tau;
    }

    auto errorStateVector() const {
        auto err = Eigen::Map<const Eigen::Matrix<uint32_t, MODEL_DOF, 1>>(error_state, MODEL_DOF);
        return err;
    }

    auto q(int idx) const {
        if (idx < MODEL_DOF) {
            return current_position[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    auto qDot(int idx) const {
        if (idx < MODEL_DOF) {
            return current_velocity[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    auto torque(int idx) const {
        if (idx < MODEL_DOF) {
            return current_torque[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    auto error(int idx) const {
        if (idx < MODEL_DOF) {
            return error_state[idx];
        } else {
            throw std::out_of_range("Joint index out of range");
        }
    }

    std::string toString() {
        std::stringstream sstr;
        for (int i = 0; i < MODEL_DOF; ++i) {
            sstr << "Motor " << std::setw(2) << std::left << i << " | ID: 0x" << std::setw(2) << std::left << std::hex << motor_id[i]
                 << std::dec << " | Joint: " << std::string(joint_name[i], 30) << " | Motor: " << std::string(motor_name[i], 30)
                 << " | Error Status: " << error_state[i] << std::dec << " | Pos: " << current_position[i]
                 << " | Vel: " << current_velocity[i] << " | Torque: " << current_torque[i] << " | Current: " << current_current[i]
                 << " | Temp: " << current_temp[i] << "\n";
        }
        return sstr.str();
    }
};

// BMS/motor initialization state. Mirrors the public DDS/ROS2 BmsInitState enum and is a
// 2-bit mask: bit0 = BMS initialized, bit1 = motor initialized (BOTH = both bits set).
enum class BmsInitState : uint8_t {
    NOT_INITIALIZED   = 0b00,  // neither
    BMS_INITIALIZED   = 0b01,  // bit0: BMS ready
    MOTOR_INITIALIZED = 0b10,  // bit1: motor ready
    BOTH_INITIALIZED  = 0b11,  // BMS | motor
};

// Single source of truth for deriving BMS/motor init state from the raw flags. Owned by the
// producer: igris_c_can for real hardware, the bridge's mujoco reader for synthesized sim data.
inline BmsInitState deriveBmsInitState(bool initialized, bool torque_on) {
    if (!initialized) {
        return BmsInitState::NOT_INITIALIZED;
    }
    return torque_on ? BmsInitState::BOTH_INITIALIZED : BmsInitState::BMS_INITIALIZED;
}

struct commonStatusData_st {
    common::Header header;
    bool isTorqueOn;
    bool isBodyPowerOn;
    bool isLegsPowerOn;
    bool isEmergencyOn;
    bool isPDUConnected;
    double PDUPowerPercent;
    bool initialized;
    uint8_t bms_init_state;  // BmsInitState: 0b00=NOT, 0b01=BMS, 0b10=MOTOR, 0b11=BOTH (set by the producer)
    uint32_t can_error[CAN_CHANNEL_NUM];
    uint32_t canErrorCount[CAN_CHANNEL_NUM];
    uint32_t pdu_version;  // PDU firmware/product version (e.g. 20260610); 0 when v1 / not detected

    commonStatusData_st()
        : header(), isTorqueOn(false), isBodyPowerOn(false), isLegsPowerOn(false), isEmergencyOn(false), isPDUConnected(false),
          PDUPowerPercent(0.0f), initialized(false), bms_init_state(0), pdu_version(0) {
        std::memset(can_error, 0, sizeof(can_error));
        std::memset(canErrorCount, 0, sizeof(canErrorCount));
    }

    void update(bool torque, bool onBodyPower, bool onLegsPower, bool estop, bool connected, float level, bool init,
                const uint32_t errors[CAN_CHANNEL_NUM], const uint32_t errorCounts[CAN_CHANNEL_NUM]) {
        isTorqueOn      = torque;
        isBodyPowerOn   = onBodyPower;
        isLegsPowerOn   = onLegsPower;
        isEmergencyOn   = estop;
        isPDUConnected  = connected;
        PDUPowerPercent = level;
        initialized     = init;
        bms_init_state  = static_cast<uint8_t>(deriveBmsInitState(init, torque));
        for (int i = 0; i < CAN_CHANNEL_NUM; ++i) {
            can_error[i]     = errors[i];
            canErrorCount[i] = errorCounts[i];
        }
    }

    std::string toString() {
        std::stringstream sstr;
        sstr << "Torque State: " << (isTorqueOn ? "ON" : "OFF") << "\n";
        sstr << "BMS On Body Power: " << (isBodyPowerOn ? "YES" : "NO") << "\n";
        sstr << "BMS On Legs Power: " << (isLegsPowerOn ? "YES" : "NO") << "\n";
        sstr << "BMS E-Stop: " << (isEmergencyOn ? "ACTIVE" : "INACTIVE") << "\n";
        sstr << "BMS Connected: " << (isPDUConnected ? "YES" : "NO") << "\n";
        sstr << "BMS Level: " << PDUPowerPercent << "%\n";
        sstr << "Initialized: " << (initialized ? "YES" : "NO") << "\n";
        sstr << "PDU Version: " << pdu_version << "\n";
        for (int i = 0; i < CAN_CHANNEL_NUM; ++i) {
            sstr << "CAN Channel " << i + 1 << " Error: 0x" << std::hex << can_error[i] << std::dec << "\n";
        }
        return sstr.str();
    }
};

struct mujocoStatusShmData_st {
    common::Header header;

    // TODO: Delete simTime and mujocoCnt later, as they are redundant with the timestamp in the header
    double simTime;
    int mujocoCnt;

    double jointPosition[robros::shm::common::MUJOCO_MAX_JOINT];
    double jointVelocity[robros::shm::common::MUJOCO_MAX_JOINT];
    double jointAcceleration[robros::shm::common::MUJOCO_MAX_JOINT];
    double jointTorque[robros::shm::common::MUJOCO_MAX_JOINT];

    double imuOrientation[4];
    double imuRpy[3];
    double imuAngularVelocity[3];
    double imuLinearAcceleration[3];
};

struct flags {
    uint8_t dataValid : 1;        // Overall data validity
    uint8_t accValid : 1;         // Accelerometer data valid
    uint8_t gyroValid : 1;        // Gyroscope data valid
    uint8_t rpyValid : 1;         // Roll/Pitch/Yaw data valid
    uint8_t quatValid : 1;        // Quaternion data valid
    uint8_t serialConnected : 1;  // Serial connection status
    uint8_t frameSync : 1;        // Frame synchronization status
    uint8_t dataFresh : 1;        // Data is fresh (within expected time window)

    uint16_t dataInvalidCount;
    uint16_t accInvalidCount;
    uint16_t gyroInvalidCount;
    uint16_t rpyInvalidCount;
    uint16_t quatInvalidCount;

    flags()
        : dataValid(0), accValid(0), gyroValid(0), rpyValid(0), quatValid(0), serialConnected(0), frameSync(0), dataFresh(0),
          dataInvalidCount(0), accInvalidCount(0), gyroInvalidCount(0), rpyInvalidCount(0), quatInvalidCount(0) {}
};

struct imuShmData_st {
    common::Header header;
    double orientation[4];
    double rpy[3];
    double angularVelocity[3];
    double linearAcceleration[3];

    float rpyRate;
    float accRate;
    float gyroRate;
    float quatRate;

    flags statusFlags;

    // TODO: Delete nowTimestampNs and lastTimestampNs later, as they are redundant with the timestamp in the header
    uint64_t nowTimestampNs;   // Timestamp in nanoseconds
    uint64_t lastTimestampNs;  // Latency timestamp in nanoseconds

    float dataRateHz;  // Actual data rate

    auto getRotationMatrix() const {
        auto quat_v = Eigen::Vector4d(orientation[0], orientation[1], orientation[2], orientation[3]);
        Eigen::Quaterniond quat(quat_v[0], quat_v[1], quat_v[2], quat_v[3]);
        return quat.normalized().toRotationMatrix();
    }

    auto orientationVector() const { return Eigen::Vector4d(orientation[0], orientation[1], orientation[2], orientation[3]); }

    auto rpyVector() const { return Eigen::Vector3d(rpy[0], rpy[1], rpy[2]); }

    auto angVelVector() const { return Eigen::Vector3d(angularVelocity[0], angularVelocity[1], angularVelocity[2]); }

    auto linAccVector() const { return Eigen::Vector3d(linearAcceleration[0], linearAcceleration[1], linearAcceleration[2]); }

    imuShmData_st()
        : orientation{0}, rpy{0}, angularVelocity{0}, linearAcceleration{0}, rpyRate(0.0f), accRate(0.0f), gyroRate(0.0f), quatRate(0.0f),
          nowTimestampNs(0), lastTimestampNs(0), dataRateHz(0.0f) {}
};

enum HandConnState : uint32_t {
    HAND_CONN_DISCONNECTED      = 0,  // USB/serial device absent or no motor responds
    HAND_CONN_CONNECTED_NO_INIT = 1,  // port open + motors reachable, awaiting manual init
    HAND_CONN_READY             = 2,  // initialized and operating
};

struct handV1StateData_st {
    common::Header header;
    double q[common::HAND_V1_JOINT_NUM];
    double dq[common::HAND_V1_JOINT_NUM];
    double tau[common::HAND_V1_JOINT_NUM];
    int16_t temperature[common::HAND_V1_JOINT_NUM];
    uint32_t status_bits[common::HAND_V1_JOINT_NUM];
    uint32_t conn_state;
    imuShmData_st imuData;
};

// Per-SHM connectivity flags for robotIoDiagnosticsData_st::shm_status (bit set = healthy).
// Meaningful bits depend on the controller variant: RealController populates CAN_MOTOR/IMU,
// MujocoSimController populates MUJOCO; BRIDGE_CONTROL/HAND bits apply to both.
enum ShmStatusBit : uint32_t {
    SHM_BRIDGE_CONTROL  = 1u << 0,
    SHM_HAND_V1         = 1u << 1,
    SHM_HAND_V1_SERVICE = 1u << 2,
    SHM_CAN_MOTOR       = 1u << 3,
    SHM_IMU             = 1u << 4,
    SHM_MUJOCO          = 1u << 5,
};

// Robot environment the active controller is driving. Mirrors the public DDS/ROS2
// RobotEnvironment enum (REAL=0, SIM=1); carried in robotControlStateData_st so the bridge
// republishes the controller's authoritative value instead of hardcoding it per transport.
enum class RobotEnvironment : uint8_t {
    REAL = 0,
    SIM  = 1,
};

// Controller -> Bridge: high-level control-state snapshot. Single-producer (control thread).
// control_mode/prev_control_mode carry robros::controller::RobotControlMode::Value as uint32_t
// to avoid a controller-package dependency from igris_c_shm.
struct robotControlStateData_st {
    common::Header header;
    uint32_t control_state;      // robros::lib::common::RobotControlState::Value
    uint32_t control_mode;       // RobotControlMode::Value
    uint32_t prev_control_mode;  // previous RobotControlMode::Value
    uint8_t initialized;         // 1 once the control loop has produced a valid state
    uint8_t need_to_go_home;     // 1 if a HOME preset must run before other motions
    uint8_t robot_environment;   // RobotEnvironment: 0=REAL, 1=SIM (set by the active controller)
    uint8_t cyclic;              // 1 if the motion-preset cyclic toggle is on (independent of playback)
    float control_hz;            // measured control-loop rate (2 s window)
    float control_avg_ms;        // average control compute time (2 s window)
    float control_max_ms;        // peak control compute time (2 s window)
    float motion_progress;       // active motion-preset playback progress 0.0~1.0 (0 when idle)
    char active_preset_id[64];   // last motion preset started via bridge (retained after stop; empty until first run)
    // TODO: Delete walk_x, this is for only igris_c_web, controlShm_st will be refacored with SPMC ring buffer for walk command input, and
    // the bridge will read from that instead of this field.
    double walk_x;   // m/s
    double walk_y;   // m/s
    double walk_wz;  // rad/s

    robotControlStateData_st()
        : header(), control_state(0), control_mode(0), prev_control_mode(0), initialized(0), need_to_go_home(0), robot_environment(0),
          cyclic(0), control_hz(0.0f), control_avg_ms(0.0f), control_max_ms(0.0f), motion_progress(0.0f), active_preset_id{}, walk_x(0.0),
          walk_y(0.0), walk_wz(0.0) {}
};

// Controller -> Bridge: input-stream rate + SHM connectivity diagnostics.
// Single-producer (update thread). Rates are computed over a 2 s window.
struct robotIoDiagnosticsData_st {
    common::Header header;
    float lowcmd_hz;
    float motor_hz;
    float imu_hz;
    float joy_hz;
    float common_hz;
    float hand_v1_hz;
    uint32_t shm_status;  // bitmask of ShmStatusBit (bit set = healthy)
    uint8_t shm_all_ok;   // 1 if the active controller's shmOk() reports all required SHMs ready
    uint8_t reserved[3];  // padding for deterministic layout across the SHM boundary

    robotIoDiagnosticsData_st()
        : header(), lowcmd_hz(0.0f), motor_hz(0.0f), imu_hz(0.0f), joy_hz(0.0f), common_hz(0.0f), hand_v1_hz(0.0f), shm_status(0),
          shm_all_ok(0), reserved{0, 0, 0} {}
};

}  // namespace robros::shm::state

#endif /* __STATE_DED39D17_AA65_48AD_844E_F2A6BA78DA93_H__ */
