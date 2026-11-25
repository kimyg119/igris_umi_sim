/**
 * @file protocol_packet_bionic.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-09-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __PROTOCOL_PACKET_BIONIC_ADB3F622_1B35_4D92_80D5_88BDFFF6A17D_H__
#define __PROTOCOL_PACKET_BIONIC_ADB3F622_1B35_4D92_80D5_88BDFFF6A17D_H__

#include "igris_c_can/can/can_handler.h"
#include "igris_c_can/motor/packet_handler.h"
#include "igris_c_can/pcan/PCANBasic.h"

#include <bitset>
#include <cstring>

namespace robros {
namespace motor {
namespace protocol {

template <typename CanMessage> class ProtocolBionic : public PacketHandler {

  private:
    static ProtocolBionic *unique_instance_;

  public:
    ProtocolBionic()  = default;
    ~ProtocolBionic() = default;

  public:
    enum class BionicError : uint16_t {
        NO_ERROR                     = 0x0000,
        MOTOR_OVERHEATING            = 0x0001,
        MOTOR_OVERCURRENT            = 0x0002,
        MOTOR_VOLTAGE_TOO_LOW        = 0x0003,
        MOTOR_ENCODER                = 0x0004,
        MOTOR_BRAKE_VOLTAGE_TOO_HIGH = 0x0006,
        DRV_DRIVE                    = 0x0007,
    };

    enum class CommunicationMode : uint8_t {
        automaticMessageMode = 1,
        QnAMode              = 2,
    };

    enum class MotorMode {
        ForcePositionHybridControl  = 0,
        ServoPositionControl        = 1,
        ServoSpeedControl           = 2,
        CurrentTorqueControlBraking = 3,
        Reserved1                   = 4,
        Reserved2                   = 5,
        Configuration               = 6,
        Query                       = 7,
    };

    enum class messageReturnStatus {
        noReturn                 = 0,
        ControlInstruction1      = 1,
        ControlInstruction2      = 2,
        ControlInstruction3      = 3,
        ConfigurationInstruction = 4,
        QueryInstruction         = 5,
    };

    enum class controlStatus : uint8_t {
        normalCurrentControl                = 0,
        TorqueControlMode                   = 1,
        VariableDampingBrakingControlMode   = 2,
        EnergyConsumptionBrakingControlMode = 3,
        RegenerativeBrakingControlMode      = 4,
        Reserved1                           = 5,
        Reserved2                           = 6,
        Reserved3                           = 7,
    };

    enum class queryCode {
        reservedCurrentlyinvalid                   = 0,
        controlLocation                            = 1,
        controlSpeed                               = 2,
        controlCurrent                             = 3,
        controlPower                               = 4,
        currentAcceleration                        = 5,
        currentFluxObservationGain                 = 6,
        currentDisturbanceCompensationCoefficient  = 7,
        currentFeedbackCompensationCoefficient     = 8,
        currentDampingCoefficient                  = 9,
        torqueCoefficient                          = 22,
        KPRangeForcePositionControlProtocol        = 23,
        KDRangeForcePositionMixingControlProtocol  = 24,
        PowerPositionMixedControlProtocolPOSRange  = 25,
        SPDRangeForcePositionControlProtocol       = 26,
        TORRangeForcePositionControlProtocol       = 27,
        CURRangeForcePositionMixingControlProtocol = 28,
        MCU_UUID                                   = 29,
        SoftwareVersionNumberAndHardwareVersion    = 30,
        CANTimeoutProtectionTime                   = 31,
        CurrentLoopKPAndKI                         = 32,
        SpeedLoopKPKAndI0                          = 33,
        PositionRingKPAndKD                        = 34,
        BrakeAndBrakeStatus                        = 37,
    };

  public:
    static ProtocolBionic *getInstance() { return unique_instance_; }

    static CanMessage RTR(uint16_t id);

    static CanMessage motorCommuncationModeSettingCommand(uint16_t id, CommunicationMode mode);

    static CanMessage motorZeroSettingCommand(uint16_t id);

    static CanMessage motorCANCommunicationIDSettingCommand(uint16_t id, uint32_t changeID);

    static CanMessage motorCANCommunicationIDResetCommand(uint16_t id);

    static CanMessage queryMotorCommuncationModeCommand(uint16_t id);

    static CanMessage queryCANCommuncationIDCommand();

    static CanMessage motorCANTimeoutTimeConfigurationCommand(uint16_t id, uint16_t value);

    // Hybrid position command. Inputs are converted to protocol-specific ranges internally.
    static CanMessage forcePositionHybridControlModeCommand(uint16_t id, float kp, float kd, float pos, float speed, float feedforward,
                                                            float maxTorque, float kpMax, float kdMax);

    // Servo position command. Position is expressed in degrees in the device protocol.
    static CanMessage servoPositionControlModeCommand(uint16_t id, float pos, float speed, float currentThreshold,
                                                      messageReturnStatus returnStatus);

    // Servo speed command. `speed` is expressed in rad/s and converted to the protocol payload internally.
    static CanMessage servoSpeedControlModeCommand(uint16_t id, float speed, float currentThreshold, messageReturnStatus returnStatus);

    // Current/torque command. The control mode selects the device behavior for the same payload shape.
    static CanMessage currentTorqueControlModeAndBrakingControlCommand(uint16_t id, controlStatus controlStatus,
                                                                       messageReturnStatus returnStatus, float currentTorque);

    // Acceleration configuration command. Input is clamped to the protocol-supported range.
    static CanMessage motorAccelerationConfigurationCommand(uint16_t id, float acceleration, uint8_t returnStatus);

    // Parameter configuration command for flux observation gain and disturbance compensation.
    static CanMessage
    ConfigurationInstructionsForMotorFluxObservationGainAndDisturbanceCompensationCoefficient(uint16_t id, float fluxGain,
                                                                                              float disturbanceCompensationCoefficient);

    // Parameter configuration command for feedback compensation and damping gain.
    static CanMessage
    motorFeedbackCompenstationCoefficientAndDampingCoefficientConfigurationCommand(uint16_t id, float motorFeedbackCompenstationCoefficient,
                                                                                   float dampingCoefficient);

    // Query command for readable motor control parameters.
    static CanMessage motorControlParameterQueryCommand(uint16_t id, ProtocolBionic::queryCode queryCode);

    static int parseFeedbackMessage(int channel, uint16_t id, uint8_t *data, uint8_t len,
                                    robros::motor::MotorMap<robros::motor::motorStatus> &motorMap);
};

}  // namespace protocol
}  // namespace motor
}  // namespace robros

#endif /* __PROTOCOL_PACKET_BIONIC_ADB3F622_1B35_4D92_80D5_88BDFFF6A17D_H__ */
