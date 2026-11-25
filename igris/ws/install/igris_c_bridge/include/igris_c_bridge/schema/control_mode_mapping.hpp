#ifndef __CONTROL_MODE_MAPPING_9C2E7B11_4A8D_4F60_B3E5_2D7A6C0F1E83_H__
#define __CONTROL_MODE_MAPPING_9C2E7B11_4A8D_4F60_B3E5_2D7A6C0F1E83_H__

#include <cstdint>
#include <optional>
#include <string_view>

#include "igris_c_shm/shm/service.h"

// Single source of truth for the BridgeCommandType <-> wire-string mapping.
//
// The wire strings double as (a) the executor command vocabulary, (b) the command
// names in igris_c_bridge/configs/schemas/commands.json, and (c) the strings logged
// across transports. The bridge's DDS, ROS2, and TCP/JSON paths all funnel through
// these two helpers so a new mode is defined in exactly one place. The ROS2 .srv and
// DDS .idl command enums mirror BridgeCommandType numerically (0..N); callers cast and
// lock that contract with static_assert at their own translation units.
//
// Bridge-internal: only igris_c_bridge translates these strings; the controller consumes
// the BridgeCommandType enum directly, so this lives here rather than in igris_c_shm.

namespace robros::bridge {

using BridgeCommandType = robros::shm::service::BridgeCommandType;

namespace detail {
struct BridgeCommandName {
    BridgeCommandType type;
    const char *name;
};

inline constexpr BridgeCommandName kBridgeCommandNames[] = {
    {BridgeCommandType::MOTION_PRESET, "MOTION_PRESET"},
    {BridgeCommandType::MOTION_PRESET_CYCLIC_TOGGLE, "MOTION_PRESET_CYCLIC_TOGGLE"},
    {BridgeCommandType::JOINT_POSITION_HOLD, "JOINT_POSITION_HOLD"},
    {BridgeCommandType::MOTION_STOP, "MOTION_STOP"},
    {BridgeCommandType::WALKMODE_ON, "WALKMODE_ON"},
    {BridgeCommandType::LOW_LEVEL_JOINT_CONTROL, "LOW_LEVEL_JOINT_CONTROL"},
    {BridgeCommandType::LOW_LEVEL_WALKMODE_ON, "LOW_LEVEL_WALKMODE_ON"},
    {BridgeCommandType::HIGH_LEVEL_WALKMODE_ON, "HIGH_LEVEL_WALKMODE_ON"},
    {BridgeCommandType::HIGH_LEVEL_JOINT_CONTROL, "HIGH_LEVEL_JOINT_CONTROL"},
    {BridgeCommandType::CUSTOM_MODE_1, "CUSTOM_MODE_1"},
    {BridgeCommandType::CUSTOM_MODE_2, "CUSTOM_MODE_2"},
    {BridgeCommandType::CUSTOM_MODE_3, "CUSTOM_MODE_3"},
    {BridgeCommandType::CUSTOM_MODE_4, "CUSTOM_MODE_4"},
    {BridgeCommandType::CUSTOM_MODE_5, "CUSTOM_MODE_5"},
    {BridgeCommandType::CUSTOM_MODE_6, "CUSTOM_MODE_6"},
    {BridgeCommandType::CUSTOM_MODE_7, "CUSTOM_MODE_7"},
    {BridgeCommandType::CUSTOM_MODE_8, "CUSTOM_MODE_8"},
    {BridgeCommandType::CUSTOM_MODE_9, "CUSTOM_MODE_9"},
    {BridgeCommandType::CUSTOM_MODE_10, "CUSTOM_MODE_10"},
};
}  // namespace detail

// BridgeCommandType -> canonical wire string. Returns nullptr for out-of-range values.
inline const char *bridgeCommandToString(BridgeCommandType type) noexcept {
    for (const auto &entry : detail::kBridgeCommandNames) {
        if (entry.type == type) {
            return entry.name;
        }
    }
    return nullptr;
}

// Convenience for the raw enum value carried by the ROS2/DDS command_type fields.
inline const char *bridgeCommandToString(uint32_t type) noexcept { return bridgeCommandToString(static_cast<BridgeCommandType>(type)); }

// Canonical wire string -> BridgeCommandType. std::nullopt if unknown.
inline std::optional<BridgeCommandType> bridgeCommandFromString(std::string_view str) noexcept {
    for (const auto &entry : detail::kBridgeCommandNames) {
        if (str == entry.name) {
            return entry.type;
        }
    }
    return std::nullopt;
}

}  // namespace robros::bridge

#endif /* __CONTROL_MODE_MAPPING_9C2E7B11_4A8D_4F60_B3E5_2D7A6C0F1E83_H__ */
