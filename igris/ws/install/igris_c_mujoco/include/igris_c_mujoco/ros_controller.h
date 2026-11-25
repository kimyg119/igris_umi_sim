#ifndef ROSCONTROLLER_H
#define ROSCONTROLLER_H

#include "igris_c_lib/kinematics/JointManager.h"
#include "igris_c_shm/shm/types.h"

#include <Eigen/Dense>
#include <deque>
#include <iostream>
#include <mujoco/mjdata.h>
#include <mujoco/mjui.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>

namespace robros::controller::igris_c {
class JointManager;
}
/**
 * @brief the data through output to the controller (CONTROL_HZ = MUJOCO_HZ /
 * MUJOCO_DATA_DECIMATION) CONTROL_HZ: the frequency of the main control thread
 * MUJOCO_HZ: the frequency of the simulation physics steps which is defined in
 * the simulation file (.xml/.mjcf) by the option/timestep (Default:
 * 500HZ/0.002s)
 *
 * Publish rate must stay 300 Hz. DECIMATION is an integer, so MUJOCO_HZ (set by
 * option/timestep in igris_c_v2.xml.xacro) must be an integer multiple of 300; change
 * timestep and DECIMATION together, and scale COMMAND_DELAY to keep ~5 ms
 * (round(0.005 * MUJOCO_HZ)). Valid combinations:
 *
 *   MUJOCO_HZ  timestep   DECIMATION  300Hz?  note
 *   3000       1/3000 s   10          yes     stable; previous default
 *   2400       1/2400 s   8           yes     stable
 *   2100       1/2100 s   7           yes     stable; current setting
 *   1800       1/1800 s   6           yes     stable (stability boundary)
 *   1500       1/1500 s   5           yes     UNSTABLE (see xacro <option>)
 *   2000       1/2000 s   6.67        no      illegal: non-integer decimation
 *
 * Larger timestep lowers closed-loop stability; keep the <connect> solref timeconst
 * >= ~2x timestep (see the <option> comment in igris_c_v2.xml.xacro).
 */
#define MUJOCO_DATA_DECIMATION 7  // 2100 Hz physics / 7 = 300 Hz publish.

#define MUJOCO_COMMAND_DELAY                                                                                                               \
    10  // dim: mujoco physics steps. Can be greater than MUJOCO_DATA_DECIMATION. ~5 ms latency at 2100 Hz (10/2100=4.76 ms).

// partial mode of real robot control's one
namespace robros::mujoco {
enum {
    IDLE             = 0,
    POSITION_CONTROL = 2,
    TORQUE_CONTROL   = 4,
    HYBRID_CONTROL   = 5,
    VELOCITY_CONTROL = 6,
};
}

using namespace std::chrono_literals;

class MujocoRos : public rclcpp::Node {
  public:
    MujocoRos();
    ~MujocoRos();

    // Expose the mujoco SHM so the physics thread can poll the
    // sim-control channel (pause/resume/reload/reset commands written by the
    // bridge_sim DDS handler).
    robros::shm::shm::mujocoShm_st *getMujocoShm() const { return mujocoShmManagerPtr_ ? mujocoShmManagerPtr_->get_shm() : nullptr; }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr base_odom_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_subscriber_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sim_command_subscriber_;

    sensor_msgs::msg::JointState joint_state_msg;
    sensor_msgs::msg::Imu imu_msg;
    nav_msgs::msg::Odometry base_odom_msg;

    bool command_received;
    int mujoco_count;
    int status_send_count;
    int command_recv_count;
    // int num_motor;
    double command_time;

    Eigen::VectorXd e_ctrl;
    Eigen::VectorXd m_ctrl;
    Eigen::VectorXd eq_pos;
    Eigen::VectorXd eq_vel;
    Eigen::VectorXd eq_acc;
    Eigen::VectorXd eq_torque;

    Eigen::VectorXi nq_actuator_idx_array;
    Eigen::VectorXi nv_actuator_idx_array;

    // member or static somewhere accessible:
    std::vector<mjtNum> base_gear;  // size = m->nu

    Eigen::VectorXd ctrl_command_temp;
    Eigen::VectorXd kp_temp;
    Eigen::VectorXd kd_temp;

    Eigen::VectorXd kp;
    Eigen::VectorXd kd;
    mjtNum *ctrl_command;

    int actuator_dof;
    bool has_last_command;
    robros::shm::command::mujocoCommandShmData_st last_command;

    struct DelayedCommand {
        int apply_mujoco_count = 0;
        robros::shm::command::mujocoCommandShmData_st command{};
    };
    std::deque<DelayedCommand> pending_commands_;
    int base_body_id_                   = -1;
    int floating_base_joint_id_         = -1;
    int floating_base_dof_adr_          = -1;
    double last_base_odom_publish_time_ = -1.0;  // MuJoCo simulation seconds; negative means not published yet.

    void initialize(const mjModel *m, mjData *d);
    void ros_sync(const mjModel *m, mjData *d);
    void publish_base_odom(const mjModel *m, const mjData *d);
    void joint_command_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

    std::shared_ptr<robros::mujoco::MujocoShmManager> mujocoShmManagerPtr_;
    std::size_t spmc_command_consumer_id_ = 0;
    uint32_t last_sim_request_seq_        = 0;  // last processed sim_service.request_seq

    std::unique_ptr<robros::controller::igris_c::JointManager> jointManager_;
    std::string mujocoParallelValue_ = "true";
    std::string mujocoModelPath_;

    const std::string &getMujocoModelPath() const { return mujocoModelPath_; }
};
#endif
