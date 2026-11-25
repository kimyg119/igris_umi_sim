/**
 * @file protocol_packet_v4.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-09-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __PROTOCOL_PACKET_V4_CB134395_7159_40A8_8B88_37345EA5C994_H__
#define __PROTOCOL_PACKET_V4_CB134395_7159_40A8_8B88_37345EA5C994_H__

#include "igris_c_can/can/can_typedef.h"
#include "igris_c_can/motor/motorTypeDef.h"
#include "igris_c_can/motor/packet_handler.h"
#include "igris_c_can/util/logger.h"

#include <bitset>
#include <cstring>
#include <iostream>

namespace robros {
namespace motor {
namespace protocol {
template <class CanMessage> class ProtocolV4 : public PacketHandler {

  private:
    static ProtocolV4 *unique_instance_;

  public:
    ProtocolV4()  = default;
    ~ProtocolV4() = default;

    static ProtocolV4 *getInstance() { return unique_instance_; }

    enum class V4Error : uint16_t {
        NO_ERROR                           = 0x0000,
        MOTOR_STALL                        = 0x0002,
        LOW_VOLTAGE                        = 0x0004,
        OVER_VOLTAGE                       = 0x0008,
        OVER_CURRENT                       = 0x0010,
        POWER_OVERRUN                      = 0x0040,
        CALIBRATION_PARAMETER_WRITING      = 0x0080,
        SPEEDING                           = 0x0100,
        COMPONENT_OVERTEMPERATURE          = 0x0800,
        MOTOR_TEMPERATURE_OVER_TEMPERATURE = 0x1000,
        ENCODER_CALIBRATION                = 0x2000,
        ENCODER_DATA                       = 0x4000
    };

    enum class functionIndex : uint8_t {
        clearMultiTurnValue                            = 1,
        CANIDFilterEnable                              = 2,
        ErrorStatusTransmissionEnable                  = 3,
        multiTurnValueSavingWhenPowerOff               = 4,
        setCANID                                       = 5,
        setMaximumPositiveAngleInPositionOperationMode = 6,
        setMaximumNegativeAngleInPositionOperationMode = 7
    };

    enum class PIDIndex : uint8_t {
        currentLoopKP  = 0x01,
        currentLoopKI  = 0x02,
        speedLoopKP    = 0x04,
        speedLoopKI    = 0x05,
        positionLoopKP = 0x07,
        positionLoopKI = 0x08,
        positionLoopKD = 0x09,
    };

    enum class advancedParameterIndex : uint8_t {
        overVoltage            = 0x13,  // with v3
        lowVoltage             = 0x14,  // with v3
        stallTime              = 0x15,  // with v3
        eBrakeStartDucyCycle   = 0x16,  // with v3
        currentSampleRate      = 0x17,  // with v3
        eBrakeHoldDucyCycle    = 0x18,  // with v3
        eBrakeMode             = 0x19,  // with v3
        encoder2AbnormalValue  = 0x3f,  // v4 only
        encoder2AbnormalSpeed  = 0x40,  // v4 only
        automaticErrorRecovery = 0x41,  // v4 only
        maxPositivePosition    = 0x1a,  // with v3
        minNegativePosition    = 0x1b,  // with v3
        positivePlanMaxAcc     = 0x1c,  // with v3
        positivePlanMaxDec     = 0x1d,  // with v3
        positivePlanMaxSpeed   = 0x1e,  // with v3
        speedPlanMaxAcc        = 0x1f,  // with v3
        speedPlanMaxDec        = 0x20,  // with v3
        motorPositionZero      = 0x21,  // with v3
        kt_out                 = 0x3e,  // v4 only
        maxTorque              = 0x46,  // v4 only
        maxCurrent             = 0x23,  // with v3
        stallCurrent           = 0x24,  // with v3
        shutdownTemp           = 0x25,  // with v3
        resumeTemp             = 0x26,  // with v3
        maxSpeed               = 0x27,  // with v3
        nominalSpeed           = 0x28,  // with v3
        enable2ndEncoder       = 0x29,  // with v3
        ratedCurrent           = 0x22,  // v4 only
        enableCANFilter        = 0x32,  // v3 only
        enableEhterCAT         = 0x3b,  // v4 only
        thermistor             = 0x3d,  // v4 only
        positionKP             = 0x08,  // with v3
        positionKI             = 0x09,  // with v3
        positionKD             = 0x0A,  // with v3
        positionFilterConst    = 0x31,  // with v3
        speedKP                = 0x06,  // with v3
        speedKI                = 0x07,  // with v3
        speedKD                = 0x2D,  // with v3
        speedFilterConst       = 0x30,  // with v3
        currentKP              = 0x04,  // with v3
        currentKI              = 0x05,  // with v3
        currentKD              = 0x2C,  // with v3
        currentFilterConst     = 0x2F,  // with v3
        currentLoopRamp        = 0x2E,  // with v3

    };

    enum class motorModel : uint8_t {
        v3 = 0,
        v4 = 1,
    };

    enum class advancedParameterIndexClass : uint8_t {
        v3only = 0,
        v4only = 1,
        v34    = 2,
    };

    static inline const std::map<advancedParameterIndex, advancedParameterIndexClass> advancedParameterClassMap = {
        {advancedParameterIndex::overVoltage, advancedParameterIndexClass::v34},
        {advancedParameterIndex::lowVoltage, advancedParameterIndexClass::v34},
        {advancedParameterIndex::stallTime, advancedParameterIndexClass::v34},
        {advancedParameterIndex::eBrakeStartDucyCycle, advancedParameterIndexClass::v34},
        {advancedParameterIndex::currentSampleRate, advancedParameterIndexClass::v34},
        {advancedParameterIndex::eBrakeHoldDucyCycle, advancedParameterIndexClass::v34},
        {advancedParameterIndex::eBrakeMode, advancedParameterIndexClass::v34},
        {advancedParameterIndex::encoder2AbnormalValue, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::encoder2AbnormalSpeed, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::automaticErrorRecovery, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::maxPositivePosition, advancedParameterIndexClass::v34},
        {advancedParameterIndex::minNegativePosition, advancedParameterIndexClass::v34},
        {advancedParameterIndex::positivePlanMaxAcc, advancedParameterIndexClass::v34},
        {advancedParameterIndex::positivePlanMaxDec, advancedParameterIndexClass::v34},
        {advancedParameterIndex::positivePlanMaxSpeed, advancedParameterIndexClass::v34},
        {advancedParameterIndex::speedPlanMaxAcc, advancedParameterIndexClass::v34},
        {advancedParameterIndex::speedPlanMaxDec, advancedParameterIndexClass::v34},
        {advancedParameterIndex::motorPositionZero, advancedParameterIndexClass::v34},
        {advancedParameterIndex::kt_out, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::maxTorque, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::maxCurrent, advancedParameterIndexClass::v34},
        {advancedParameterIndex::stallCurrent, advancedParameterIndexClass::v34},
        {advancedParameterIndex::shutdownTemp, advancedParameterIndexClass::v34},
        {advancedParameterIndex::resumeTemp, advancedParameterIndexClass::v34},
        {advancedParameterIndex::maxSpeed, advancedParameterIndexClass::v34},
        {advancedParameterIndex::nominalSpeed, advancedParameterIndexClass::v34},
        {advancedParameterIndex::enable2ndEncoder, advancedParameterIndexClass::v34},
        {advancedParameterIndex::ratedCurrent, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::enableCANFilter, advancedParameterIndexClass::v3only},
        {advancedParameterIndex::enableEhterCAT, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::thermistor, advancedParameterIndexClass::v4only},
        {advancedParameterIndex::positionKP, advancedParameterIndexClass::v34},
        {advancedParameterIndex::positionKI, advancedParameterIndexClass::v34},
        {advancedParameterIndex::positionKD, advancedParameterIndexClass::v34},
        {advancedParameterIndex::positionFilterConst, advancedParameterIndexClass::v34},
        {advancedParameterIndex::speedKP, advancedParameterIndexClass::v34},
        {advancedParameterIndex::speedKI, advancedParameterIndexClass::v34},
        {advancedParameterIndex::speedKD, advancedParameterIndexClass::v34},
        {advancedParameterIndex::speedFilterConst, advancedParameterIndexClass::v34},
        {advancedParameterIndex::currentKP, advancedParameterIndexClass::v34},
        {advancedParameterIndex::currentKI, advancedParameterIndexClass::v34},
        {advancedParameterIndex::currentKD, advancedParameterIndexClass::v34},
        {advancedParameterIndex::currentFilterConst, advancedParameterIndexClass::v34},
        {advancedParameterIndex::currentLoopRamp, advancedParameterIndexClass::v34}};

    static constexpr uint16_t multiCommandId = 0x280 - 0x140;

    static CanMessage RTR(uint16_t id);
    static CanMessage readAdvancedParameterCommand(uint16_t id, advancedParameterIndex parameterIndex);
    static CanMessage writeAdvancedParameterCommand(uint16_t id, advancedParameterIndex parameterIndex, float value);
    static CanMessage saveAdvanedParameterToROMCommand(uint16_t id);
    static CanMessage ReadPIDParameterCommand(uint16_t id, PIDIndex pidIndex);
    static CanMessage writePIDParameterRAMCommand(uint16_t id, PIDIndex pidIndex, float value);
    static CanMessage writePIDParameterROMCommand(uint16_t id, PIDIndex pidIndex, float value);
    static CanMessage writeEncoderMultiTurnValue2ROMasMotorZeroCommand(uint16_t id, int32_t offset);
    static CanMessage systemResetCommand(uint16_t id);
    static CanMessage writeCurrentMultiTurnPositionEncoder2ROMasMotorZeroCommand(uint16_t id);
    static CanMessage readMultiTurnEncoderPositionDataCommand(uint16_t id);
    static CanMessage readMultiTurnEncoderOriginPositonDataCommand(uint16_t id);
    static CanMessage readMultiTurnEncoderZeroOffsetDataCommand(uint16_t id);
    static CanMessage readSingleTurnEncoderCommand(uint16_t id);
    static CanMessage readMultiTurnAngleCommand(uint16_t id);

    // Reads one ASCII chunk of the motor model string. Call with 1, 2, 3 and concatenate responses.
    static CanMessage motorModelReadingCommand(uint16_t id, uint8_t asciiNum);
    static CanMessage systemSoftwareVersionDataReadCommand(uint16_t id);
    static CanMessage forceControlPositionClosedLoopCommand(uint16_t id, float pos, float speedLimit, float maxTorque);
    static CanMessage forceControlPositionClosedLoopCommand2(uint16_t id, float pos, float speedLimit, float maxTorque);
    static CanMessage torqueClosedLoopControlCommand(uint16_t id, float torque);
    static CanMessage torqueClosedLoopControlCommand2(uint16_t id, float torque);
    static CanMessage speedClosedLoopControlCommand(uint16_t id, float speed);
    static CanMessage absolutePositionClosedLoopControlCommand(uint16_t id, float pos, float speedLimit);
    static CanMessage absolutePositionClosedLoopControlCommand2(uint16_t id, float pos, float speedLimit);
    static CanMessage incrementalPositionClosedLoopControlCommand(uint16_t id, float pos, float speedLimit);
    static CanMessage readMotorStatusCommand1(uint16_t id);
    static CanMessage readMotorStatusCommand2(uint16_t id);
    static CanMessage readMotorStatusCommand3(uint16_t id);
    static CanMessage functionControlCommand(uint16_t id, functionIndex index, uint32_t value);
    static CanMessage IDChangeCommand(uint16_t id, uint32_t new_id);
    static CanMessage CANIDSettingCommand(uint16_t newId);
    static CanMessage CANIDReadCommand();
    static CanMessage motorShutdownCommand(uint16_t id);
    static CanMessage motorStopCommand(uint16_t id);
    static CanMessage communicationInterruptProtectionTimeSettingCommand(uint16_t id, uint32_t timeMs);
    static CanMessage motionModeControlCommand(uint16_t id, float kp, float kd, float pos, float speed, float feedforwrad);
    static CanMessage motionModeControlCommand2(uint16_t id, float kp, float kd, float pos, float speed, float feedforwrad,
                                                float maxTorque);
    static int parseFeedbackMessage(int channel, uint16_t id, uint8_t *data, uint8_t len,
                                    robros::motor::MotorMap<robros::motor::motorStatus> &statusMap);
};

}  // namespace protocol
}  // namespace motor
}  // namespace robros

#endif /* __PROTOCOL_PACKET_V4_CB134395_7159_40A8_8B88_37345EA5C994_H__ */
