/**
 * @file jointManager.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __JOINTMANAGER_A789D3FD_6244_43FE_8FB0_8C2B44486A31_H__
#define __JOINTMANAGER_A789D3FD_6244_43FE_8FB0_8C2B44486A31_H__

#include "igris_c/common/robot_define.h"
#include "igris_c_lib/kinematics/link.h"
#include "igris_c_lib/kinematics/safe_set.h"
#include "igris_c_shm/logger/logger.h"
#include "igris_c_shm/shm/command.h"
#include "igris_c_shm/shm/state.h"

namespace robros::controller {

namespace igris_c {
static constexpr char joint_names[MODEL_DOF][50] = {
    "waist_yaw",     "waist_roll",       "waist_pitch",     "l_hip_pitch",    "l_hip_roll",    "l_hip_yaw",    "l_knee_pitch",
    "l_ankle_pitch", "l_ankle_roll",     "r_hip_pitch",     "r_hip_roll",     "r_hip_yaw",     "r_knee_pitch", "r_ankle_pitch",
    "r_ankle_roll",  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow_pitch", "l_wrist_yaw",  "l_wrist_roll",
    "l_wrist_pitch", "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch", "r_wrist_yaw",  "r_wrist_roll",
    "r_wrist_pitch", "neck_yaw",         "neck_pitch",
};

class JointManager {
  public:
    explicit JointManager(const std::string &logger_name);
    ~JointManager();

    robros::shm::logger::Logger &getLogger() const { return logger_; }

    int initialize(const std::string &parallel_links_path);
    int initialize(const std::string &parallel_links_path, const std::string &safe_set_root_dir);

    int motor2Joint(const robros::shm::state::motorStatusData_st &from, robros::shm::command::RobotCommand &to);

    int motor2Joint(const robros::shm::state::motorStatusData_st &from, robros::shm::state::jointStatusData_st &to);

    int motorCommand2Joint(const robros::shm::state::motorStatusData_st &motor_status, const robros::shm::command::RobotCommand &from,
                           robros::shm::command::RobotCommand &to,
                           const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes);

    int motorCommand2Joint(const robros::shm::state::motorStatusData_st &motor_status, const robros::shm::command::RobotCommand &from,
                           robros::shm::state::jointStatusData_st &to,
                           const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes);

    int joint2Motor(const robros::shm::state::jointStatusData_st &from, robros::shm::state::motorStatusData_st &to,
                    const std::array<ParallelLinkIKSolverMode, kParallelLinkModeCount> &parallel_ik_solver_modes,
                    const std::array<ParallelLinkSafetyMode, kParallelLinkModeCount> &parallel_safety_modes);

    int jointCommand2Motor(const robros::shm::state::motorStatusData_st &motor_status, const robros::shm::command::RobotCommand &from,
                           robros::shm::command::RobotCommand &to,
                           const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes,
                           const std::array<ParallelLinkIKSolverMode, kParallelLinkModeCount> &parallel_ik_solver_modes,
                           const std::array<ParallelLinkSafetyMode, kParallelLinkModeCount> &parallel_safety_modes,
                           const std::array<bool, kParallelLinkModeCount> &parallel_pd_gain_mapping_enabled);

    int jointCommand2Motor(const robros::shm::state::motorStatusData_st &motor_status, const robros::shm::command::RobotCommand &from,
                           robros::shm::command::RobotCommand &to,
                           const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes,
                           const std::array<ParallelLinkIKSolverMode, kParallelLinkModeCount> &parallel_ik_solver_modes,
                           const std::array<ParallelLinkSafetyMode, kParallelLinkModeCount> &parallel_safety_modes,
                           const std::array<ParallelLinkSoftSafeSetMode, kParallelLinkModeCount> &parallel_soft_safeset_modes,
                           const std::array<bool, kParallelLinkModeCount> &parallel_pd_gain_mapping_enabled,
                           const std::array<bool, kParallelLinkModeCount> &parallel_ff_torque_compensation_enabled);

    int jointCommand2Motor(const robros::shm::state::motorStatusData_st &motor_status, const robros::shm::command::RobotCommand &from,
                           robros::shm::command::motorCommandData_st &to,
                           const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes,
                           const std::array<ParallelLinkIKSolverMode, kParallelLinkModeCount> &parallel_ik_solver_modes,
                           const std::array<ParallelLinkSafetyMode, kParallelLinkModeCount> &parallel_safety_modes,
                           const std::array<bool, kParallelLinkModeCount> &parallel_pd_gain_mapping_enabled);

    int jointCommand2Motor(const robros::shm::state::motorStatusData_st &motor_status, const robros::shm::command::RobotCommand &from,
                           robros::shm::command::motorCommandData_st &to,
                           const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes,
                           const std::array<ParallelLinkIKSolverMode, kParallelLinkModeCount> &parallel_ik_solver_modes,
                           const std::array<ParallelLinkSafetyMode, kParallelLinkModeCount> &parallel_safety_modes,
                           const std::array<ParallelLinkSoftSafeSetMode, kParallelLinkModeCount> &parallel_soft_safeset_modes,
                           const std::array<bool, kParallelLinkModeCount> &parallel_pd_gain_mapping_enabled,
                           const std::array<bool, kParallelLinkModeCount> &parallel_ff_torque_compensation_enabled);

    std::array<int, 2> GetParallelLinkJointIdPair(ParallelLinkModeIndex mode_index) const;
    const std::array<Parallel4BarLink, kParallelLinkModeCount> &GetParallelFourBarLinks() const { return parallel_fourbar_links_; }

    enum class WristSide { LEFT, RIGHT };

  private:
    bool wrist_fourbar_inverse_kinematics(double input_angle, double *output_angle, WristSide side);

    bool wrist_fourbar_forward_kinematics(double input_angle, double *output_angle, WristSide side, double *vel_ratio,
                                          double *torque_ratio);

    Parallel4BarLink &GetParallelFourBarLinkReference(ParallelLinkModeIndex mode_index);

    const Parallel4BarLink &GetParallelFourBarLinkReference(ParallelLinkModeIndex mode_index) const;

    void Joint2MotorGains(const std::array<ParallelLinkCommandMode, kParallelLinkModeCount> &parallel_kinematic_modes,
                          const std::array<bool, kParallelLinkModeCount> &parallel_pd_gain_mapping_enabled,
                          Eigen::VectorQd desired_joint_kp, Eigen::VectorQd desired_joint_kd, Eigen::VectorQd &desired_motor_kp,
                          Eigen::VectorQd &desired_motor_kd);

    const SafeSet2D *GetParallelSafeSet(ParallelLinkModeIndex mode_index, ParallelLinkCommandMode command_mode,
                                        ParallelLinkSoftSafeSetMode soft_mode) const;

    void ApplyParallelLinkSafety(Eigen::VectorQd &desired_position, ParallelLinkModeIndex mode_index, ParallelLinkCommandMode command_mode,
                                 ParallelLinkSafetyMode safety_mode, ParallelLinkSoftSafeSetMode soft_mode) const;

  private:
    std::array<Parallel4BarLink, kParallelLinkModeCount> parallel_fourbar_links_;

    std::array<double, 2> left_wrist_zero_pose_velocity_ratios_{1.0, 1.0};   // [front, back]
    std::array<double, 2> right_wrist_zero_pose_velocity_ratios_{1.0, 1.0};  // [front, back]
    std::array<double, 2> left_wrist_zero_pose_torque_ratios_{1.0, 1.0};     // [front, back]
    std::array<double, 2> right_wrist_zero_pose_torque_ratios_{1.0, 1.0};    // [front, back]

    std::array<Eigen::Matrix2d, kParallelLinkModeCount> parallel_link_zero_pose_constant_jacobians_{
        Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Identity(),
        Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Identity(),
    };

    std::array<double, 2> left_wrist_forward_velocity_ratios_{1.0, 1.0};   // [front, back]
    std::array<double, 2> left_wrist_forward_torque_ratios_{1.0, 1.0};     // [front, back]
    std::array<double, 2> right_wrist_forward_velocity_ratios_{1.0, 1.0};  // [front, back]
    std::array<double, 2> right_wrist_forward_torque_ratios_{1.0, 1.0};    // [front, back]
    bool verbose_;
    robros::shm::logger::Logger &logger_;

    SafeSetManager ssm_;
    const SafeSet2D *ss2d_left_wrist_joints_  = nullptr;
    const SafeSet2D *ss2d_right_wrist_joints_ = nullptr;
    const SafeSet2D *ss2d_left_ankle_joints_  = nullptr;
    const SafeSet2D *ss2d_right_ankle_joints_ = nullptr;
    const SafeSet2D *ss2d_waist_joints_       = nullptr;

    const SafeSet2D *ss2d_left_wrist_motors_  = nullptr;
    const SafeSet2D *ss2d_right_wrist_motors_ = nullptr;
    const SafeSet2D *ss2d_left_ankle_motors_  = nullptr;
    const SafeSet2D *ss2d_right_ankle_motors_ = nullptr;
    const SafeSet2D *ss2d_waist_motors_       = nullptr;

    const SafeSet2D *ss2d_soft_left_wrist_joints_  = nullptr;
    const SafeSet2D *ss2d_soft_right_wrist_joints_ = nullptr;
    const SafeSet2D *ss2d_soft_left_ankle_joints_  = nullptr;
    const SafeSet2D *ss2d_soft_right_ankle_joints_ = nullptr;
    const SafeSet2D *ss2d_soft_waist_joints_       = nullptr;

    const SafeSet2D *ss2d_soft_left_wrist_motors_  = nullptr;
    const SafeSet2D *ss2d_soft_right_wrist_motors_ = nullptr;
    const SafeSet2D *ss2d_soft_left_ankle_motors_  = nullptr;
    const SafeSet2D *ss2d_soft_right_ankle_motors_ = nullptr;
    const SafeSet2D *ss2d_soft_waist_motors_       = nullptr;

    std::array<ParallelLinkSoftSafeSetMode, kParallelLinkModeCount> parallel_soft_safeset_modes_{
        ParallelLinkSoftSafeSetMode::HARD, ParallelLinkSoftSafeSetMode::HARD, ParallelLinkSoftSafeSetMode::HARD,
        ParallelLinkSoftSafeSetMode::HARD, ParallelLinkSoftSafeSetMode::HARD};

    std::array<ParallelLinkCommandMode, kParallelLinkModeCount> parallel_kinematic_modes_{
        ParallelLinkCommandMode::JOINT_SPACE, ParallelLinkCommandMode::JOINT_SPACE, ParallelLinkCommandMode::JOINT_SPACE,
        ParallelLinkCommandMode::JOINT_SPACE, ParallelLinkCommandMode::JOINT_SPACE,
    };

    std::array<bool, kParallelLinkModeCount> parallel_pd_gain_mapping_enabled_{
        false, false, false, false, false,
    };
};
}  // namespace igris_c
}  // namespace robros::controller

#endif /* __JOINTMANAGER_A789D3FD_6244_43FE_8FB0_8C2B44486A31_H__ */
