/**
 * @file command.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __COMMAND_C0B86ADB_416B_4C57_8AE1_9E07A1E774A2_H__
#define __COMMAND_C0B86ADB_416B_4C57_8AE1_9E07A1E774A2_H__

#include "igris_c/common/robot_define.h"
#include "igris_c_shm/shm/common.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace robros::shm::command {

enum control_mode : uint32_t {
    IDLE                     = 0,
    RESERVED_0               = 1,
    POSITION_CONTROL         = 2,
    CURRENT_CONTROL          = 3,
    TORQUE_CONTROL           = 4,
    HYBRID_CONTROL           = 5,
    RESERVED_1               = 6,
    POSITION_CURRENT_CONTROL = 7,
};

struct motorCommandData_st {
    common::Header header;
    uint32_t motor_id[MODEL_DOF];
    uint32_t control_mode[MODEL_DOF];   // e.g., POSITION_CONTROL
    double target_position[MODEL_DOF];  // radians or degrees
    double target_torque[MODEL_DOF];    // Nm
    double max_torque[MODEL_DOF];       // Nm // deprecated later
    double target_current[MODEL_DOF];   // A
    double target_speed[MODEL_DOF];     // rad/s

    double target_kp[MODEL_DOF];  //
    double target_kd[MODEL_DOF];  //

    uint64_t timestamp_us[MODEL_DOF];  //
    uint64_t cnt;                      // For synchronization

    void update(int idx, uint32_t motorId, uint32_t controlMode, double targetPosition, double targetTorque, double targetCurrent,
                double targetSpeed, double targetKp, double targetKd) {
        motor_id[idx]        = motorId;
        control_mode[idx]    = controlMode;
        target_position[idx] = targetPosition;
        target_torque[idx]   = targetTorque;
        target_current[idx]  = targetCurrent;
        target_speed[idx]    = targetSpeed;
        target_kp[idx]       = targetKp;
        target_kd[idx]       = targetKd;
        // timestamp_us[idx]    = robros::util::microTick();
    }

    std::string toString() {
        std::stringstream sstr;
        for (int i = 0; i < MODEL_DOF; ++i) {
            sstr << "Motor " << std::setw(2) << std::left << i << " | ID: 0x" << std::hex << motor_id[i] << std::dec
                 << " | Pos: " << target_position[i] << " | Torque: " << target_torque[i] << " | Current: " << target_current[i]
                 << " | Speed: " << target_speed[i] << " | Kp: " << target_kp[i] << " | Kd: " << target_kd[i] << "\n";
        }
        return sstr.str();
    }
};

struct mujocoCommandShmData_st {
    double commandTime;

    double targetTorque[common::MUJOCO_MAX_JOINT];
    double targetPosition[common::MUJOCO_MAX_JOINT];
    double targetVelocity[common::MUJOCO_MAX_JOINT];

    double targetKp[common::MUJOCO_MAX_JOINT];
    double targetKd[common::MUJOCO_MAX_JOINT];

    int32_t controlMode[common::MUJOCO_MAX_JOINT];
};

enum class KinematicMode : uint32_t {
    MS  = 0,
    PJS = 1,
};

struct RobotCommand {
    using VectorQd = Eigen::Matrix<double, MODEL_DOF, 1>;
    common::Header header;
    KinematicMode
        kinematicModes[common::PARALLEL_JOINT_NUM];  // For each kinematic chain (waist, left_ankle, right_ankle, left_wrist, right_wrist)
    robros::shm::command::control_mode controlMode[MODEL_DOF];
    VectorQd q;
    VectorQd qDot;
    VectorQd torque;
    VectorQd kp;
    VectorQd kd;

    RobotCommand() {
        header.update();
        for (auto &mode : kinematicModes) {
            mode = KinematicMode::MS;
        }
        for (auto &mode : controlMode) {
            mode = control_mode::IDLE;
        }
        q.setZero();
        qDot.setZero();
        torque.setZero();
        kp.setZero();
        kd.setZero();
    }

    void setAllKinematicModes(KinematicMode mode) {
        for (auto &kinematic_mode : kinematicModes) {
            kinematic_mode = mode;
        }
    }

    bool hasKinematicMode(KinematicMode mode) const {
        return std::any_of(std::begin(kinematicModes), std::end(kinematicModes),
                           [mode](const KinematicMode &kinematic_mode) { return kinematic_mode == mode; });
    }

    bool allKinematicModesAre(KinematicMode mode) const {
        return std::all_of(std::begin(kinematicModes), std::end(kinematicModes),
                           [mode](const KinematicMode &kinematic_mode) { return kinematic_mode == mode; });
    }
};
// Walk velocity (base twist) command from bridge to controller.
// Mirrors ROS 2 geometry_msgs/Twist, robot base convention (REP-103):
//   linear_x  forward (+),  linear_y  left (+),  angular_z  yaw CCW (+)
//   linear_z / angular_x / angular_y unused, kept 0.
struct bridgeWalkData_st {
    common::Header header;
    double linear_x;
    double linear_y;
    double linear_z;
    double angular_x;
    double angular_y;
    double angular_z;
};

// Low-level motor command from bridge to controller
struct bridgeLowcmdMotor_st {
    common::Header header;
    uint16_t id;
    float q;
    float dq;
    float tau;
    float kp;
    float kd;
};

struct bridgeLowcmdData_st {
    common::Header header;
    bridgeLowcmdMotor_st motors[MODEL_DOF];
    uint32_t kinematic_mode[common::PARALLEL_JOINT_NUM];
};

struct handV1Command_st {
    common::Header header;
    uint16_t id[common::HAND_V1_JOINT_NUM];
    double q[common::HAND_V1_JOINT_NUM];
    double dq[common::HAND_V1_JOINT_NUM];
    double tau[common::HAND_V1_JOINT_NUM];
    double kp[common::HAND_V1_JOINT_NUM];
    double kd[common::HAND_V1_JOINT_NUM];
};

}  // namespace robros::shm::command

#endif /* __COMMAND_C0B86ADB_416B_4C57_8AE1_9E07A1E774A2_H__ */
