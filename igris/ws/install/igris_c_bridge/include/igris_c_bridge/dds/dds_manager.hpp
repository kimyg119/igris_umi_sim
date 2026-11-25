#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "igris_c_bridge/data_store.hpp"
#include "igris_c_bridge/types.hpp"
#include "igris_c_sdk/channel_factory.hpp"
#include "igris_c_sdk/publisher.hpp"
#include "igris_c_sdk/subscriber.hpp"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::dds {

class DdsManager {
  public:
    explicit DdsManager(std::shared_ptr<DataStore> data_store);
    ~DdsManager();

    bool start(int domain_id, const std::string &ns);
    void stop();

    // Context setters (for service callbacks)
    void setControlContext(std::shared_ptr<ControlContext> ctx);
    void setSettingsContext(std::shared_ptr<SettingsContext> ctx);

    // Environment (SIM/REAL) — must be set before start() so the publishing
    // thread reports the correct value in RobotState.
    void setEnvironment(igris_c::msg::dds::RobotEnvironment env);

    // Sim-only: register the Mujoco sim-control service (DDS request -> SHM
    // sim_command channel). Call after start() and after the mujoco SHM is
    // open. No-op for real builds.
    bool enableMujocoSimService(robros::shm::shm::mujocoShm_st *mujoco_shm);

    // LowCmd access (received via DDS, forwarded to controller via SHM)
    bool getLowCmd(LowCmd &cmd_out);
    bool waitAndGetLowCmd(LowCmd &cmd_out, std::atomic<bool> &running);
    void notifyLowCmdWaiters();
    bool isLowCmdAvailable() const;
    void updateLowCmdFromCurrentState(const sensor_msgs::msg::JointState &joint_state);

    // WalkTwist access (received via DDS rt/cmd_vel, forwarded to controller via SHM)
    bool waitAndGetWalkTwist(WalkTwist &cmd_out, std::atomic<bool> &running);
    void notifyWalkTwistWaiters();

  private:
    void publishingThread();
    void lowcmdCallback(const LowCmd &cmd);
    void walktwistCallback(const WalkTwist &cmd);

    // Service request callbacks
    void bmsInitRequestCallback(const BmsInitCmd &cmd);
    void torqueRequestCallback(const TorqueCmd &cmd);
    void controlModeCommandRequestCallback(const igris_c::msg::dds::ControlModeCommandRequest &cmd);
    void mujocoSimRequestCallback(const igris_c::msg::dds::MujocoSimCmd &cmd);
    void handInitRequestCallback(const HandInitRequest &cmd);

    // Shared data store
    std::shared_ptr<DataStore> data_store_;

    // Contexts for service callbacks
    std::shared_ptr<ControlContext> control_ctx_;
    std::shared_ptr<SettingsContext> settings_ctx_;

    // DDS Publishers
    std::unique_ptr<igris_c_sdk::Publisher<LowState>> lowstate_pub_;
    std::unique_ptr<igris_c_sdk::Publisher<BmsState>> bmsstate_pub_;
    std::unique_ptr<igris_c_sdk::Publisher<RobotState>> robotstate_pub_;

    // State messages (headers auto-increment seq across publishes)
    LowState low_state_;
    BmsState bms_state_;
    RobotState robot_state_;

    // DDS Subscribers
    std::unique_ptr<igris_c_sdk::Subscriber<LowCmd>> lowcmd_sub_;
    std::unique_ptr<igris_c_sdk::Subscriber<WalkTwist>> walktwist_sub_;

    // LowCmd buffer
    mutable std::mutex lowcmd_mtx_;
    std::condition_variable lowcmd_cv_;
    LowCmd latest_lowcmd_;
    bool lowcmd_available_{false};

    // WalkTwist buffer
    mutable std::mutex walktwist_mtx_;
    std::condition_variable walktwist_cv_;
    WalkTwist latest_walktwist_;
    bool walktwist_available_{false};

    // Environment (SIM/REAL) — published in RobotState every cycle.
    igris_c::msg::dds::RobotEnvironment environment_{igris_c::msg::dds::RobotEnvironment::ROBOT_ENV_REAL};

    // Sim-only: mujoco SHM handle for sim-control writes (nullptr in real).
    robros::shm::shm::mujocoShm_st *mujoco_shm_{nullptr};

    // Service Request Subscribers
    std::unique_ptr<igris_c_sdk::Subscriber<BmsInitCmd>> bms_init_req_sub_;
    std::unique_ptr<igris_c_sdk::Subscriber<TorqueCmd>> torque_req_sub_;
    std::unique_ptr<igris_c_sdk::Subscriber<igris_c::msg::dds::ControlModeCommandRequest>> control_mode_req_sub_;
    std::unique_ptr<igris_c_sdk::Subscriber<igris_c::msg::dds::MujocoSimCmd>> mujoco_sim_req_sub_;

    // Service Response Publishers
    std::unique_ptr<igris_c_sdk::Publisher<ServiceResponse>> bms_init_res_pub_;
    std::unique_ptr<igris_c_sdk::Publisher<ServiceResponse>> torque_res_pub_;
    std::unique_ptr<igris_c_sdk::Publisher<ServiceResponse>> control_mode_res_pub_;
    std::unique_ptr<igris_c_sdk::Publisher<ServiceResponse>> mujoco_sim_res_pub_;

    // HandInit server-side: external SDK clients (cyclonedds/python) publish a request here; the
    // bridge routes it to the hand node over SHM (setHandInitAsync) and publishes the response,
    // mirroring the bms_init/torque/control_mode services.
    std::unique_ptr<igris_c_sdk::Subscriber<HandInitRequest>> hand_init_req_sub_;
    std::unique_ptr<igris_c_sdk::Publisher<ServiceResponse>> hand_init_res_pub_;

    // Publishing thread
    std::thread publishing_thread_;
    std::atomic<bool> publishing_running_{false};
};

}  // namespace robros::bridge::dds
