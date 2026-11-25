#ifndef __SERVICE_B49B3B02_439E_4B7D_BDEB_B0BA4377709A_H__
#define __SERVICE_B49B3B02_439E_4B7D_BDEB_B0BA4377709A_H__

#include "igris_c_shm/shm/common.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace robros::shm::service {

struct OnOffRequest_st {
    bool onOff;  // true for ON, false for OFF
};

struct OnOffResponse_st {
    bool success;
    char message[100];

    void set(bool isSuccess, const std::string &msg) {
        success = isSuccess;
        std::memcpy(message, msg.c_str(), std::min(msg.size(), sizeof(message) - 1));
    }
};

// Bridge <-> Controller command type
enum class BridgeCommandType : uint32_t {
    MOTION_PRESET = 0,
    MOTION_PRESET_CYCLIC_TOGGLE,
    JOINT_POSITION_HOLD,
    MOTION_STOP,
    WALKMODE_ON,
    LOW_LEVEL_JOINT_CONTROL,
    LOW_LEVEL_WALKMODE_ON,
    HIGH_LEVEL_WALKMODE_ON,
    HIGH_LEVEL_JOINT_CONTROL,
    CUSTOM_MODE_1,
    CUSTOM_MODE_2,
    CUSTOM_MODE_3,
    CUSTOM_MODE_4,
    CUSTOM_MODE_5,
    CUSTOM_MODE_6,
    CUSTOM_MODE_7,
    CUSTOM_MODE_8,
    CUSTOM_MODE_9,
    CUSTOM_MODE_10
};

// Service request: Bridge -> Controller
struct BridgeServiceRequest_st {
    common::Header header;
    BridgeCommandType command_type;
    char preset_id[64];  // valid when command_type == MOTION_PRESET
    bool is_cyclic;      // valid when command_type == MOTION_PRESET_CYCLIC_TOGGLE
};

// Service response: Controller -> Bridge
struct BridgeServiceResponse_st {
    common::Header header;
    bool success;
    char message[256];
};

// Mujoco sim-control commands (bridge_sim -> mujoco), wrapped in
// ServiceSharedMemory inside mujocoShm_st.
enum class MujocoSimCommandType : uint32_t {
    SIM_NONE   = 0,
    SIM_PAUSE  = 1,
    SIM_RESUME = 2,
    SIM_RELOAD = 3,
    SIM_RESET  = 4,
};

struct mujocoSimRequest_st {
    MujocoSimCommandType command_type;
};

struct mujocoSimResponse_st {
    bool success;
    char message[100];

    void set(bool isSuccess, const std::string &msg) {
        success             = isSuccess;
        const std::size_t n = std::min(msg.size(), sizeof(message) - 1);
        std::memcpy(message, msg.c_str(), n);
        message[n] = '\0';
    }
};

}  // namespace robros::shm::service

#endif /* __SERVICE_B49B3B02_439E_4B7D_BDEB_B0BA4377709A_H__ */
