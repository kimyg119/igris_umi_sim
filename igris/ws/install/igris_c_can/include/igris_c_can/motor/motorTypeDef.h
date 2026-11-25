/**
 * @file motorTypeDef.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-09-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __MOTORTYPEDEF_C014AC93_4DD4_468C_9945_24A7A7583F70_H__
#define __MOTORTYPEDEF_C014AC93_4DD4_468C_9945_24A7A7583F70_H__

#include <cstdint>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
//
#include "yaml-cpp/yaml.h"

namespace robros {
namespace motor {

static const char *v3_version_2025_1 = "2025030301";  // year month day build
static const char *v3_version_2025_2 = "2025102701";
static const char *v3_version_2025_3 = "2025112201";
static const char *v4_version_2025_1 = "2025070202";
static const char *v4_version_2025_2 = "2025102701";

static const std::vector<std::map<std::string, std::string>> igris_c_motor_versions_1 = {
    {{"v3", v3_version_2025_1}, {"v4", v4_version_2025_1}}};

static const std::vector<std::map<std::string, std::string>> igris_c_motor_versions_2 = {
    {{"v3", v3_version_2025_2}, {"v4", v4_version_2025_2}}};

enum MotorControlProtocolVersion : uint8_t {
    PROTOCOL_VERSION_1 = 1,
    PROTOCOL_VERSION_2 = 2,
};

enum MotorError : uint32_t {
    // clang-format off
    NO_ERROR                                 = 0x00000000,

    INITIALIZE_ERROR                         = 0x00000001,
    PACKET_NOT_RECEIVED_ERROR                = 0x00000002,

    ROBOT_LEVEL_MASK                         = 0x00000003,

    /* After the current reaches the stall current, the speed is
    very low and continues for a period of time. 
    It indicates that the motor load is too large. */
    MOTOR_STALL_ERROR                        = 0x00000004,
    /* The power input is lower than the set undervoltage
    value */
    LOW_VOLTAGE_ERROR                        = 0x00000008,
    /* The power input is higher than the set overvoltage
    value */
    OVER_VOLTAGE_ERROR                       = 0x00000010,
    /* The software detects that the motor current exceeds 
    the limit value,and there may be short circuit,phase
    loss,loss of control,motor damage,etc. */
    OVER_CURRENT_ERROR                       = 0x00000020,
    /* If the input current of the power supply exceeds the
    limit value,there may be a situation where the load is
    too large or the speed is too high. */
    POWER_OVERRUN_ERROR                      = 0x00000040,
    /* Failed to write parameters causing parameter loss. */
    CALIBRATION_PARAMETER_WRITING_ERROR      = 0x00000080,
    /* The motor running speed exceeds the limit
    value,there may be over pressure and drag use. */
    SPEEDING_ERROR                           = 0x00000100,
    /* If the motor temperature exceeds the set value,there
    may be short circuit,parameter error,and long-term
    overload use. */
    COMPONENT_OVERTEMPERATURE_ERROR          = 0x00000200,
    MOTOR_TEMPERATURE_OVER_TEMPERATURE_ERROR = 0x00000400,
    /* The encoder calibration result deviates too much
    from the standard value. */
    ENCODER_CALIBRATION_ERROR                = 0x00000800,
    /* */
    ENCODER_DATA_ERROR                       = 0x00001000,
    /* */
    MOTOR_BRAKE_VOLTAGE_TOO_HIGH_ERROR       = 0x00002000,
    /* */
    DRV_DRIVE_ERROR                          = 0x00004000,
    // clang-format on
};

struct PIDGains {
    float p_gain;
    float i_gain;
    float d_gain;

    PIDGains() : p_gain(0.0f), i_gain(0.0f), d_gain(0.0f) {}
    PIDGains(float p, float i, float d) : p_gain(p), i_gain(i), d_gain(d) {}
};

enum class MotorClass {
    UNKNOWN,
    RMD_X2_P28_7,
    X6_P8_V3,
    RMD_X6_P8_8,
    RMD_X6_P20_60,
    BIONIC_X8_90,
    RMD_X8_P20_120,
    BIONIC_X12_150,
};

inline std::string motorClassToString(MotorClass motorClass) {
    switch (motorClass) {
    case MotorClass::RMD_X2_P28_7:
        return "RMD_X2_P28_7";
    case MotorClass::X6_P8_V3:
        return "X6_P8_V3";
    case MotorClass::RMD_X6_P8_8:
        return "RMD_X6_P8_8";
    case MotorClass::RMD_X6_P20_60:
        return "RMD_X6_P20_60";
    case MotorClass::BIONIC_X8_90:
        return "BIONIC_X8_90";
    case MotorClass::RMD_X8_P20_120:
        return "RMD_X8_P20_120";
    case MotorClass::BIONIC_X12_150:
        return "BIONIC_X12_150";
    case MotorClass::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

struct motorConfig {
    int channel;
    uint16_t id;
    std::string name;
    std::string protocol;
    std::string className;
    MotorClass motorClass;

    float limit[2];
    float acceleration;

    float offset;
    int urdf_axis_dir;
    float torque_constant;
    float max_torque;

    // Optional v4 advanced parameters written to motor ROM at init. The presence
    // flags let init skip a parameter entirely when its key is absent in the config.
    float over_voltage    = 0.0f;
    bool has_over_voltage = false;
    float stall_time      = 0.0f;
    bool has_stall_time   = false;

    float interruptTime;

    PIDGains positionLoopGains;
    PIDGains speedLoopGains;
    PIDGains currentLoopGains;

    float motionModeKpScale;
    float motionModeKdScale;

    float motionModeKpMax;
    float motionModeKdMax;

    motorConfig() = default;

    motorConfig(YAML::Node node) {
        channel         = node["can_channel"].as<int>();
        id              = node["id"].as<uint16_t>();
        name            = node["name"].as<std::string>();
        className       = node["motor_class"].as<std::string>();
        protocol        = node["protocol"].as<std::string>();
        offset          = node["offset"].as<float>();
        urdf_axis_dir   = node["urdf_axis_dir"].as<int>();
        torque_constant = node["torque_constant"].as<float>();
        max_torque      = node["max_torque"].as<float>();
        // Optional v4 advanced params: read like the other keys, but only when present
        // (absent key -> left unset so init skips that parameter).
        if (node["over_voltage"]) {
            over_voltage     = node["over_voltage"].as<float>();
            has_over_voltage = true;
        }
        if (node["stall_time"]) {
            stall_time     = node["stall_time"].as<float>();
            has_stall_time = true;
        }
        // Default to 0.5s when the key is absent to stay compatible with older configs.
        interruptTime = node["interrupt_time"] ? node["interrupt_time"].as<float>() : 0.5f;

        positionLoopGains = PIDGains(node["position_kp"].as<float>(), node["position_ki"].as<float>(), node["position_kd"].as<float>());
        speedLoopGains    = PIDGains(node["speed_kp"].as<float>(), node["speed_ki"].as<float>(), node["speed_kd"].as<float>());
        currentLoopGains  = PIDGains(node["current_kp"].as<float>(), node["current_ki"].as<float>(), node["current_kd"].as<float>());

        motionModeKpScale = node["motion_mode_kp_scale"].as<float>();
        motionModeKdScale = node["motion_mode_kd_scale"].as<float>();

        motionModeKpMax = node["motion_mode_kp_max"].as<float>();
        motionModeKdMax = node["motion_mode_kd_max"].as<float>();

        if (className == "RMD_X2_P28_7")
            motorClass = MotorClass::RMD_X2_P28_7;
        else if (className == "X6_P8_V3")
            motorClass = MotorClass::X6_P8_V3;
        else if (className == "RMD_X6_P8_8")
            motorClass = MotorClass::RMD_X6_P8_8;
        else if (className == "RMD_X6_P20_60")
            motorClass = MotorClass::RMD_X6_P20_60;
        else if (className == "BIONIC_X8_90")
            motorClass = MotorClass::BIONIC_X8_90;
        else if (className == "RMD_X8_P20_120")
            motorClass = MotorClass::RMD_X8_P20_120;
        else if (className == "BIONIC_X12_150")
            motorClass = MotorClass::BIONIC_X12_150;
        else
            motorClass = MotorClass::UNKNOWN;
    }

    void setPIDGains(const PIDGains &posGains, const PIDGains &speedGains, const PIDGains &currentGains) {
        positionLoopGains = posGains;
        speedLoopGains    = speedGains;
        currentLoopGains  = currentGains;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "MotorConfig - Channel: " << channel << ", ID: " << id << ", Name: " << name << ", Protocol: " << protocol
           << ", Class: " << motorClassToString(motorClass) << ", Acceleration: " << acceleration << ", Offset: " << offset;
        return ss.str();
    }

    std::string toLogString() const { return toString(); }
};

struct motorStatus {
    std::string name;
    std::string model;

    int channel;
    uint16_t id;
    std::string firmwareVersion;
    MotorClass motorClass;

    uint32_t errorState;
    uint16_t controlMode;
    uint16_t packetRate;
    uint16_t receiveCount;

    float nowPos;
    float nowSpeed;
    float nowCurrent;
    float nowTorque;

    float acceleration;

    float targetPos;
    float targetTorque;
    float targetCurrent;
    float targetSpeed;

    float targetKp;
    float targetKd;

    float motorTemperature;
    float MOSTemperature;
    float voltage;

    PIDGains positionLoopGains;
    PIDGains speedLoopGains;
    PIDGains currentLoopGains;

    motorStatus() = default;

    motorStatus(motorConfig config) {
        name             = config.name;
        model            = "Unknown";
        channel          = config.channel;
        id               = config.id;
        motorClass       = config.motorClass;
        controlMode      = 0;
        errorState       = 0;
        nowPos           = 0.0f;
        nowSpeed         = 0.0f;
        nowCurrent       = 0.0f;
        nowTorque        = 0.0f;
        targetPos        = 0.0f;
        targetTorque     = 0.0f;
        targetCurrent    = 0.0f;
        targetSpeed      = 0.0f;
        motorTemperature = 0.0f;
        MOSTemperature   = 0.0f;
        voltage          = 0.0f;
        receiveCount     = 0;
        packetRate       = 0;
    }

    motorStatus(std::string name, int ch, uint16_t id, MotorClass number)
        : name(name), model("Unknown"), channel(ch), id(id), motorClass(number), controlMode(0), errorState(0), nowPos(0.0f),
          nowSpeed(0.0f), nowCurrent(0.0f), nowTorque(0.0f), targetPos(0.0f), targetTorque(0.0f), targetCurrent(0.0f), targetSpeed(0.0f),
          motorTemperature(0.0f), MOSTemperature(0.0f), voltage(0.0f), receiveCount(0), packetRate(0) {}

    std::string toBasicString() const {
        std::stringstream ss;
        ss << "Channel: " << channel << ", ID: " << std::setw(3) << std::left << id << ", Name: " << std::setw(25) << std::left << name
           << ", Model: " << std::setw(22) << std::left << model << ", Firmware: " << std::setw(10) << std::left << firmwareVersion
           << ", controlMode: " << controlMode << ", errorState: " << errorState;
        return ss.str();
    }

    std::string toPIDString() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4) << "Channel: " << channel << ", ID: " << std::setw(3) << std::left << id
           << ", Name: " << std::setw(25) << std::left << name << "\n"
           << std::setw(70) << " " << "Position PID: " << positionLoopGains.p_gain << ", " << positionLoopGains.i_gain << ", "
           << positionLoopGains.d_gain << ", Speed PID: " << speedLoopGains.p_gain << ", " << speedLoopGains.i_gain << ", "
           << speedLoopGains.d_gain << ", Current PID: " << currentLoopGains.p_gain << ", " << currentLoopGains.i_gain << ", "
           << currentLoopGains.d_gain;
        return ss.str();
    }

    // std::string toPIDString() const {
    //     std::stringstream ss;
    //     ss << std::fixed << std::setprecision(6) << "Motor PID Gains - Channel: " << channel << ", ID: " << id << ", Name: " << name <<
    //     "\n"
    //        << "positionLoopKp: " << positionLoopGains.p_gain << "\npositionLoopKi: " << positionLoopGains.i_gain
    //        << "\npositionLoopKd: " << positionLoopGains.d_gain << "\n"
    //        << "speedLoopKp: " << speedLoopGains.p_gain << "\nspeedLoopKi: " << speedLoopGains.i_gain
    //        << "\nspeedLoopKd: " << speedLoopGains.d_gain << "\n"
    //        << "currentLoopKp: " << currentLoopGains.p_gain << "\ncurrentLoopKi: " << currentLoopGains.i_gain
    //        << "\ncurrentLoopKd: " << currentLoopGains.d_gain;
    //     return ss.str();
    // }

    std::string toLogString() const {
        std::stringstream sstr;
        sstr << std::fixed << std::setprecision(4) << "ID: " << std::setw(3) << std::left << id << ", Mode: " << std::setw(2) << std::left
             << controlMode << ", Pos: " << std::setw(8) << std::left << nowPos << ", Speed: " << std::setw(8) << std::left << nowSpeed
             << ", Cur: " << std::setw(8) << std::left << nowCurrent << ", Tor: " << std::setw(8) << std::left << nowTorque
             << ", TPos: " << std::setw(8) << std::left << targetPos << ", TTor: " << std::setw(8) << std::left << targetTorque
             << ", TSpd: " << std::setw(8) << std::left << targetSpeed << ", TCur: " << std::setw(8) << std::left << targetCurrent
             << ", Temp: " << std::setw(6) << std::left << motorTemperature << ", Err: " << fmt::format("0x{:08X}", errorState) << std::dec
             << std::nouppercase << std::setfill(' ');
        return sstr.str();
    }

    std::string toString() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << "Motor Status - Channel: " << channel << ", ID: " << std::setw(3) << std::left << id
           << ", Name: " << std::setw(25) << std::left << name << ", Model: " << model << ", Firmware: " << firmwareVersion
           << ", controlMode: " << controlMode << ", errorState: " << errorState << std::endl;
        ss << ", NowPos: " << nowPos << ", NowSpeed: " << nowSpeed << ", NowCur: " << nowCurrent << ", NowTor: " << nowTorque
           << ", TargetPos: " << targetPos << ", TargetTor: " << targetTorque << ", TargetCur: " << targetCurrent
           << ", TargetSpeed: " << targetSpeed << ", Acce: " << acceleration << ", MotorTemp: " << motorTemperature
           << ", MOSTemp: " << MOSTemperature;
        return ss.str();
    }
};

template <typename T> class MotorMap {
  public:
    MotorMap() {
        static_assert(std::is_class<T>::value, "T must be a class or struct type.");
        static_assert(std::is_member_object_pointer<decltype(&T::channel)>::value, "T must have a member named 'channel'.");
        static_assert(std::is_member_object_pointer<decltype(&T::id)>::value, "T must have a member named 'id'.");
    }
    ~MotorMap() {}

    int addData(T data) {
        for (const auto &motor : motorVector_) {
            if (motor.channel == data.channel && motor.id == data.id) {
                return -1;  // Duplicate entry
            }
        }
        motorVector_.push_back(data);
        return 0;  // Success
    }

    T &getRef(int channel, uint16_t id) {
        for (auto &motor : motorVector_) {
            if (motor.channel == channel && motor.id == id) {
                return motor;
            }
        }
        throw std::runtime_error("Motor status not found for channel: " + std::to_string(channel) + ", ID: " + std::to_string(id));
    }

    T *getPtr(int id) {
        for (auto &motor : motorVector_) {
            if (motor.id == id) {
                return &motor;
            }
        }
        return nullptr;
    }

    auto begin() { return motorVector_.begin(); }
    auto end() { return motorVector_.end(); }
    auto begin() const { return motorVector_.begin(); }
    auto end() const { return motorVector_.end(); }

    std::vector<T> &getData() { return motorVector_; }

    size_t size() const { return motorVector_.size(); }

    std::string toLogString() const {
        std::stringstream sstr;
        for (const auto &motor : motorVector_) {
            sstr << "\n" << motor.toLogString();
        }
        return sstr.str();
    }

  private:
    std::vector<T> motorVector_;
};
}  // namespace motor
}  // namespace robros

#endif /* __MOTORTYPEDEF_C014AC93_4DD4_468C_9945_24A7A7583F70_H__ */
