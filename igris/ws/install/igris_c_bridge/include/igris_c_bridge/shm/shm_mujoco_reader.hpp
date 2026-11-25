#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "igris_c_bridge/bridge_state.hpp"
#include "igris_c_bridge/data_store.hpp"
#include "igris_c_bridge/dds/dds_manager.hpp"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::shm {

// Reads sensor data from Mujoco shared memory and feeds into DataStore.
// Also forwards DDS LowCmd to controller via SPSC shared memory.
// Used in simulation mode only.
class ShmMujocoReader {
  public:
    ShmMujocoReader(std::shared_ptr<DataStore> data_store, std::shared_ptr<BridgeState> state, std::shared_ptr<dds::DdsManager> dds_manager,
                    robros::shm::shm::mujocoShm_st *mujoco_shm, robros::shm::shm::canShm_st *can_shm,
                    robros::shm::shm::controlShm_st *spsc_shm, robros::shm::shm::handV1Shm_st *hand_v1_spmc);

    bool start();
    void stop();

  private:
    void mujocoFeederThread();
    void handV1FeederThread();
    void lowcmdForwardingThread();
    void twistForwardingThread();

    static constexpr const char *kSpmcConsumerName = "bridge";

    std::shared_ptr<DataStore> data_store_;
    std::shared_ptr<BridgeState> state_;
    std::shared_ptr<dds::DdsManager> dds_manager_;
    robros::shm::shm::mujocoShm_st *mujoco_shm_;
    robros::shm::shm::canShm_st *can_shm_;
    robros::shm::shm::controlShm_st *spsc_shm_;
    robros::shm::shm::handV1Shm_st *hand_v1_spmc_;

    std::thread mujoco_feeder_thread_;
    std::thread hand_v1_feeder_thread_;
    std::thread lowcmd_thread_;
    std::thread twist_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace robros::bridge::shm
