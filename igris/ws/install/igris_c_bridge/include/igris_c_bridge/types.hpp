#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "igris_c_lib/util/schema.h"
#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/types.hpp"

namespace robros::bridge {

using json = nlohmann::json;

// DDS message type aliases
using LowState   = igris_c::msg::dds::LowState;
using BmsState   = igris_c::msg::dds::BmsState;
using LowCmd     = igris_c::msg::dds::LowCmd;
using WalkTwist  = igris_c::msg::dds::WalkTwist;
using RobotState = igris_c::msg::dds::RobotState;

using BmsInitCmd                = igris_c::msg::dds::BmsInitCmd;
using TorqueCmd                 = igris_c::msg::dds::TorqueCmd;
using ControlModeCommandRequest = igris_c::msg::dds::ControlModeCommandRequest;
using HandInitRequest           = igris_c::msg::dds::HandInitRequest;
using ServiceResponse           = igris_c::msg::dds::ServiceResponse;

// Bridge-specific robot status (to avoid dependency on controller)
struct BridgeRobotStatus {
    bool bms_is_on_body_power_ = false;
    bool bms_is_on_legs_power_ = false;
    bool bms_estop_            = false;
    bool bms_connected_        = false;
    bool is_initialized_       = false;
    bool is_torque_on_         = false;
    uint8_t bms_init_state_    = 0;  // BmsInitState, produced upstream (igris_c_can / sim reader)

    double battery_percent_ = 0.0;  // 0..100 [%]; upstream PDUPowerPercent

    std::array<float, 31> motor_temp_{};
    std::array<int, 31> motor_error_state_{};
};

struct RequestCommon : robros::lib::util::SchemaStruct<RequestCommon> {
    std::optional<std::string> requestId;
    std::string functionType;  // Server-internal: populated from commands.json schema
    std::string commandType;
    json commandBody;

    // Convenience accessor for requestId (returns "" if absent)
    const std::string &reqId() const {
        static const std::string kEmpty;
        return requestId.has_value() ? *requestId : kEmpty;
    }

    static constexpr auto schemaFields() {
        return std::make_tuple(robros::lib::util::makeField("requestId", &RequestCommon::requestId, false),
                               robros::lib::util::makeField("functionType", &RequestCommon::functionType, false),
                               robros::lib::util::makeField("commandType", &RequestCommon::commandType, true),
                               robros::lib::util::makeField("commandBody", &RequestCommon::commandBody, false));
    }
};

// Interfaces

class IControlExecutor {
  public:
    virtual ~IControlExecutor() = default;

    virtual void submitAsync(const std::string &requestId, const std::string &commandType, const json &commandBody,
                             std::function<void(bool ok, const std::string &error_code, const std::string &msg, const json &data)> cb) = 0;

    virtual json statusJson() = 0;
};

class ISettingsService {
  public:
    virtual ~ISettingsService() = default;

    virtual void setTorqueAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) = 0;

    virtual void setMotorAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) = 0;

    virtual void setBmsPowerAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) = 0;

    virtual void setMotorInitAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) = 0;

    // Hand initialization via the SHM hand_init_service (internal path is SHM-only, not DDS).
    virtual void setHandInitAsync(bool enable, std::function<void(bool ok, const std::string &msg)> cb) = 0;
};

// Forward declarations
struct BridgeState;
namespace dds {
class DdsManager;
}

// Contexts

struct ControlContext {
    std::shared_ptr<IControlExecutor> exec;
    std::shared_ptr<BridgeState> state;
};

struct MonitoringContext {
    std::function<json()> statusGetter;
    std::string motions_dir;
    std::string robot_namespace;
    int dds_domain_id       = 0;
    int tcp_port            = 9000;
    std::string tcp_bind_ip = "0.0.0.0";
};

struct SettingsContext {
    std::shared_ptr<ISettingsService> settings;
    std::shared_ptr<BridgeState> state;
    std::shared_ptr<dds::DdsManager> dds_manager;  // For DDS-forwarded settings (HAND_INIT)
};

struct BridgeConfig : robros::lib::util::SchemaStruct<BridgeConfig> {
    int tcp_port            = 9000;
    std::string tcp_bind_ip = "0.0.0.0";
    int dds_domain_id       = 0;
    std::optional<std::string> dds_network_interface;
    std::string robot_namespace;

    static constexpr auto schemaFields() {
        return std::make_tuple(robros::lib::util::makeField("tcp_port", &BridgeConfig::tcp_port, false),
                               robros::lib::util::makeField("tcp_bind_ip", &BridgeConfig::tcp_bind_ip, false),
                               robros::lib::util::makeField("dds_domain_id", &BridgeConfig::dds_domain_id, false),
                               robros::lib::util::makeField("dds_network_interface", &BridgeConfig::dds_network_interface, false),
                               robros::lib::util::makeField("namespace", &BridgeConfig::robot_namespace, false));
    }
};

}  // namespace robros::bridge
