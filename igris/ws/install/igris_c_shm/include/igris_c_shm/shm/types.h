/**
 * @file types.h
 * @author colson (colson@robros.co.kr)
 * @brief Shared memory types for robot communication
 * @version 0.1
 * @date 2026-06-04
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __TYPES_D503F777_6DCE_4D98_9625_7D66011C793F_H__
#define __TYPES_D503F777_6DCE_4D98_9625_7D66011C793F_H__

#include "igris_c_shm/buffer/SpmcRingBuffer.h"
#include "igris_c_shm/buffer/SpscRingBuffer.h"
#include "igris_c_shm/service/service.h"
#include "igris_c_shm/shm/command.h"
#include "igris_c_shm/shm/service.h"
#include "igris_c_shm/shm/sharedMemory.h"
#include "igris_c_shm/shm/state.h"

namespace robros::shm::shm {

namespace detail {
constexpr std::size_t kRingCapacity          = 64;
constexpr std::size_t kSpmcMaxConsumer       = 8;
constexpr std::size_t kMujocoRingCapacity    = 8;
constexpr std::size_t kMujocoSpmcMaxConsumer = 8;
}  // namespace detail

struct canShm_st {
    std::atomic<uint32_t> magic;

    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::command::motorCommandData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        joint_command_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::motorStatusData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        motor_status_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::jointStatusData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        joint_status_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::commonStatusData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        common_status_buffer;

    robros::shm::service::ServiceSharedMemory<robros::shm::service::OnOffRequest_st, robros::shm::service::OnOffResponse_st> bms_service;
    robros::shm::service::ServiceSharedMemory<robros::shm::service::OnOffRequest_st, robros::shm::service::OnOffResponse_st> torque_service;
    robros::shm::service::ServiceSharedMemory<robros::shm::service::OnOffRequest_st, robros::shm::service::OnOffResponse_st>
        initialize_service;
};

struct imuShm_st {
    std::atomic<uint32_t> magic;

    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::imuShmData_st, detail::kRingCapacity, detail::kSpmcMaxConsumer>
        imu_buffer;
};

struct mujocoShm_st {
    std::atomic<uint32_t> magic;

    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::mujocoStatusShmData_st, detail::kMujocoRingCapacity,
                                                     detail::kMujocoSpmcMaxConsumer>
        mujoco_status_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::motorStatusData_st, detail::kMujocoRingCapacity,
                                                     detail::kMujocoSpmcMaxConsumer>
        motor_status_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::jointStatusData_st, detail::kMujocoRingCapacity,
                                                     detail::kMujocoSpmcMaxConsumer>
        mujoco_joint_status_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::command::mujocoCommandShmData_st, detail::kMujocoRingCapacity,
                                                     detail::kMujocoSpmcMaxConsumer>
        mujoco_command_buffer;
    robros::shm::service::ServiceSharedMemory<robros::shm::service::mujocoSimRequest_st, robros::shm::service::mujocoSimResponse_st>
        sim_service;
};

// Controller-owned segment: the controller creates it; the bridge opens it and writes commands.
// Hosts all bridge -> controller command paths: the SPSC fire-and-forget buffers (command/walk/
// lowcmd) and the embedded request/response command service (ctrl_service), which the controller
// (server) serves in place and the bridge (client) attaches to after opening the segment.
struct controlShm_st {
    std::atomic<uint32_t> magic;

    robros::shm::buffer::SpscRingBuffer<robros::shm::service::BridgeServiceRequest_st, detail::kRingCapacity> command_buffer;
    robros::shm::buffer::SpscRingBuffer<robros::shm::command::bridgeWalkData_st, detail::kRingCapacity> walk_buffer;
    robros::shm::buffer::SpscRingBuffer<robros::shm::command::bridgeLowcmdData_st, detail::kRingCapacity> lowcmd_buffer;
    robros::shm::service::ServiceSharedMemory<robros::shm::service::BridgeServiceRequest_st, robros::shm::service::BridgeServiceResponse_st>
        ctrl_service;
};

struct handV1Shm_st {
    std::atomic<uint32_t> magic;

    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::handV1StateData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        hand_v1_state_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::command::handV1Command_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        hand_v1_command_buffer;
    robros::shm::service::ServiceSharedMemory<robros::shm::service::OnOffRequest_st, robros::shm::service::OnOffResponse_st>
        hand_init_service;
};

// Controller-owned segment: pure state output. The controller creates/writes (SPMC, multi-consumer:
// control_state from the control thread, io_diag from the update thread); the bridge and web open
// and read it for monitoring. Command input lives in controlShm_st, not here.
struct robotStateShm_st {
    std::atomic<uint32_t> magic;

    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::robotControlStateData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        control_state_buffer;
    robros::shm::buffer::SpmcBroadcastDropRingBuffer<robros::shm::state::robotIoDiagnosticsData_st, detail::kRingCapacity,
                                                     detail::kSpmcMaxConsumer>
        io_diag_buffer;
};

}  // namespace robros::shm::shm

namespace robros::can {
static constexpr uint32_t CAN_SPMC_MAGIC    = 0x43414E00;  // '0CAN'
static constexpr uint32_t CAN_SERVICE_MAGIC = 0x53455256;  // 'SERV' (embedded bms/torque/initialize services)
static constexpr const char *CAN_SHM_NAME   = "/igris_c_can_spmc";

using MotorShmManager = robros::shm::shm::SharedMemoryManager<robros::shm::shm::canShm_st>;
}  // namespace robros::can

namespace robros::imu {
static constexpr uint32_t IMU_SPMC_MAGIC  = 0x494D5500;  // '0IMU'
static constexpr const char *IMU_SHM_NAME = "/igris_c_imu_shm";

using ImuShmManager = robros::shm::shm::SharedMemoryManager<robros::shm::shm::imuShm_st>;
}  // namespace robros::imu

namespace robros::mujoco {
static constexpr uint32_t MUJOCO_SHM_MAGIC         = 0x4D4A434F;  // 'MJCO'
static constexpr uint32_t MUJOCO_SIM_SERVICE_MAGIC = 0x53455256;  // 'SERV'
static constexpr const char *MUJOCO_SHM_NAME       = "/mujoco_shm";

using MujocoShmManager = robros::shm::shm::SharedMemoryManager<robros::shm::shm::mujocoShm_st>;

}  // namespace robros::mujoco

namespace robros::hand::v1 {
static constexpr uint32_t HAND_V1_SHM_MAGIC     = 0x48414E44;  // 'HAND'
static constexpr uint32_t HAND_V1_SERVICE_MAGIC = 0x53455256;  // 'SERV' (embedded hand_init_service)
static constexpr const char *HAND_V1_SHM_NAME   = "/igris_c_hand_v1_shm";

using HandV1ShmManager = robros::shm::shm::SharedMemoryManager<robros::shm::shm::handV1Shm_st>;
}  // namespace robros::hand::v1

namespace robros::controller {
// State-output segment (pure state; controller writes, bridge/web read).
static constexpr uint32_t ROBOT_STATE_SHM_MAGIC   = 0x52425354;  // 'RBST'
static constexpr const char *ROBOT_STATE_SHM_NAME = "/igris_c_robot_state_spmc";

// Command-input segment (controller owns/creates, bridge opens/writes). Hosts the SPSC
// command/walk/lowcmd buffers and the embedded ctrl_service (CONTROL_SERVICE_MAGIC).
static constexpr uint32_t CONTROL_SHM_MAGIC     = 0x42435351;  // 'BCSP'
static constexpr uint32_t CONTROL_SERVICE_MAGIC = 0x53455256;  // 'SERV' (embedded ctrl_service)
static constexpr const char *CONTROL_SHM_NAME   = "/igris_c_control_shm";

using RobotStateShmManager = robros::shm::shm::SharedMemoryManager<robros::shm::shm::robotStateShm_st>;
using ControlShmManager    = robros::shm::shm::SharedMemoryManager<robros::shm::shm::controlShm_st>;
}  // namespace robros::controller

#endif /* __TYPES_D503F777_6DCE_4D98_9625_7D66011C793F_H__ */
