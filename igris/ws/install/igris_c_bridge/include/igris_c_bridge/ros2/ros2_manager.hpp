#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>
#include <thread>

#include "igris_c_bridge/data_store.hpp"
#include "igris_c_bridge/types.hpp"
#include "igris_c_sdk/msg/bms_state.hpp"
#include "igris_c_sdk/msg/low_cmd.hpp"
#include "igris_c_sdk/msg/low_state.hpp"
#include "igris_c_sdk/msg/robot_state.hpp"
#include "igris_c_sdk/msg/walk_twist.hpp"
#include "igris_c_sdk/srv/bms_init_cmd.hpp"
#include "igris_c_sdk/srv/control_mode_command_request.hpp"
#include "igris_c_sdk/srv/hand_init_request.hpp"
#include "igris_c_sdk/srv/mujoco_sim_cmd.hpp"
#include "igris_c_sdk/srv/torque_cmd.hpp"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::ros2 {

class Ros2Manager {
  public:
    Ros2Manager(rclcpp::Node::SharedPtr node, std::shared_ptr<DataStore> data_store, robros::shm::shm::controlShm_st *spsc_shm);
    ~Ros2Manager();

    void setControlContext(std::shared_ptr<ControlContext> ctx);
    void setSettingsContext(std::shared_ptr<SettingsContext> ctx);
    void setEnvironment(igris_c::msg::dds::RobotEnvironment env);

    bool start(const std::string &robot_namespace = "");
    void stop();
    bool enableMujocoSimService(robros::shm::shm::mujocoShm_st *mujoco_shm);

    std::string rosNamespace() const { return resolveTopic("rt"); }

  private:
    using LowCmdMsg     = igris_c_sdk::msg::LowCmd;
    using LowStateMsg   = igris_c_sdk::msg::LowState;
    using BmsStateMsg   = igris_c_sdk::msg::BmsState;
    using RobotStateMsg = igris_c_sdk::msg::RobotState;

    static constexpr std::size_t kDof       = 31;
    static constexpr auto kRobotStatePeriod = std::chrono::milliseconds(100);
    static constexpr auto kBmsStatePeriod   = std::chrono::milliseconds(333);
    static constexpr auto kServiceTimeout   = std::chrono::milliseconds(3000);
    static constexpr auto kPollPeriod       = std::chrono::milliseconds(5);

    static bool waitBoolResult(std::function<void(std::function<void(bool, const std::string &)>)> starter, bool &ok, std::string &msg);
    static const char *controlCommandTypeToString(uint8_t command_type);

    std::string resolveTopic(const std::string &topic) const;

    void publishingThread();
    void lowcmdCallback(const LowCmdMsg &msg);
    void twistCallback(const igris_c_sdk::msg::WalkTwist &msg);

    LowStateMsg buildLowState(const DataStore::JointSnapshot &joint_snap, const DataStore::MotorSnapshot &motor_snap,
                              const BridgeRobotStatus &robot_status_snap, bool robot_status_available,
                              const DataStore::ImuSnapshot &imu_snap);
    BmsStateMsg buildBmsState(const BridgeRobotStatus &robot_status_snap, bool robot_status_available);
    RobotStateMsg buildRobotState();
    sensor_msgs::msg::Imu buildStandardImu(const DataStore::ImuSnapshot &imu_snap);
    sensor_msgs::msg::JointState buildStandardJointState(const sensor_msgs::msg::JointState &snapshot_state);

    void stamp(igris_c_sdk::msg::Header &header, uint32_t seq, const std::string &frame_id = "");
    void stampServiceResponse(igris_c_sdk::msg::Header &header);

    void handleBmsInit(const std::shared_ptr<igris_c_sdk::srv::BmsInitCmd::Request> request,
                       std::shared_ptr<igris_c_sdk::srv::BmsInitCmd::Response> response);
    void handleTorque(const std::shared_ptr<igris_c_sdk::srv::TorqueCmd::Request> request,
                      std::shared_ptr<igris_c_sdk::srv::TorqueCmd::Response> response);
    void handleControlMode(const std::shared_ptr<igris_c_sdk::srv::ControlModeCommandRequest::Request> request,
                           std::shared_ptr<igris_c_sdk::srv::ControlModeCommandRequest::Response> response);
    void handleHandInit(const std::shared_ptr<igris_c_sdk::srv::HandInitRequest::Request> request,
                        std::shared_ptr<igris_c_sdk::srv::HandInitRequest::Response> response);
    void handleMujocoSim(const std::shared_ptr<igris_c_sdk::srv::MujocoSimCmd::Request> request,
                         std::shared_ptr<igris_c_sdk::srv::MujocoSimCmd::Response> response);

    template <typename ResponseT>
    void fillResponse(ResponseT &response, const std::string &request_id, bool success, const std::string &message, int32_t error_code) {
        stampServiceResponse(response.header);
        response.request_id = request_id;
        response.success    = success;
        response.message    = message;
        response.error_code = error_code;
    }

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<DataStore> data_store_;
    robros::shm::shm::controlShm_st *spsc_shm_{nullptr};
    robros::shm::shm::mujocoShm_st *mujoco_shm_{nullptr};

    std::shared_ptr<ControlContext> control_ctx_;
    std::shared_ptr<SettingsContext> settings_ctx_;
    igris_c::msg::dds::RobotEnvironment environment_{igris_c::msg::dds::RobotEnvironment::ROBOT_ENV_REAL};

    std::string robot_namespace_;

    rclcpp::Publisher<LowStateMsg>::SharedPtr low_state_pub_;
    rclcpp::Publisher<BmsStateMsg>::SharedPtr bms_state_pub_;
    rclcpp::Publisher<RobotStateMsg>::SharedPtr robot_state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr motor_state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Subscription<LowCmdMsg>::SharedPtr low_command_sub_;
    rclcpp::Subscription<igris_c_sdk::msg::WalkTwist>::SharedPtr twist_command_sub_;

    rclcpp::CallbackGroup::SharedPtr command_callback_group_;
    rclcpp::CallbackGroup::SharedPtr service_callback_group_;

    rclcpp::Service<igris_c_sdk::srv::BmsInitCmd>::SharedPtr bms_init_srv_;
    rclcpp::Service<igris_c_sdk::srv::TorqueCmd>::SharedPtr torque_srv_;
    rclcpp::Service<igris_c_sdk::srv::ControlModeCommandRequest>::SharedPtr control_mode_srv_;
    rclcpp::Service<igris_c_sdk::srv::HandInitRequest>::SharedPtr hand_init_srv_;
    rclcpp::Service<igris_c_sdk::srv::MujocoSimCmd>::SharedPtr mujoco_sim_srv_;

    std::thread publishing_thread_;
    std::atomic<bool> publishing_running_{false};
    std::atomic<uint32_t> low_state_seq_{0};
    std::atomic<uint32_t> bms_state_seq_{0};
    std::atomic<uint32_t> robot_state_seq_{0};
    std::atomic<uint32_t> service_response_seq_{0};
};

}  // namespace robros::bridge::ros2
