#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "igris_c_bridge/bridge_state.hpp"
#include "igris_c_bridge/data_store.hpp"
#include "igris_c_bridge/dds/dds_manager.hpp"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::shm {

// Reads sensor data from CAN and IMU SPMC shared memory and feeds into DataStore.
// Also forwards DDS LowCmd to controller via SPSC shared memory.
class ShmSensorReader {
  public:
    ShmSensorReader(std::shared_ptr<DataStore> data_store, std::shared_ptr<BridgeState> state, std::shared_ptr<dds::DdsManager> dds_manager,
                    robros::shm::shm::canShm_st *can_spmc, robros::shm::shm::imuShm_st *imu_spmc, robros::shm::shm::controlShm_st *spsc_shm,
                    robros::shm::shm::handV1Shm_st *hand_v1_spmc);

    bool start();
    void stop();

  private:
    void canFeederThread();
    void imuFeederThread();
    void handV1FeederThread();
    void lowcmdForwardingThread();
    void twistForwardingThread();

    static constexpr const char *kSpmcConsumerName = "bridge";

    std::shared_ptr<DataStore> data_store_;
    std::shared_ptr<BridgeState> state_;
    std::shared_ptr<dds::DdsManager> dds_manager_;
    robros::shm::shm::canShm_st *can_spmc_;
    robros::shm::shm::imuShm_st *imu_spmc_;
    robros::shm::shm::controlShm_st *spsc_shm_;
    robros::shm::shm::handV1Shm_st *hand_v1_spmc_;

    std::thread can_feeder_thread_;
    std::thread imu_feeder_thread_;
    std::thread hand_v1_feeder_thread_;
    std::thread lowcmd_thread_;
    std::thread twist_thread_;
    std::atomic<bool> running_{false};
};

}  // namespace robros::bridge::shm
