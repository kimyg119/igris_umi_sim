#ifndef __PROTOCOL_PACKET_PDU_V2_AC98289C_AC63_4C33_8B9C_2C137F39E508_H__
#define __PROTOCOL_PACKET_PDU_V2_AC98289C_AC63_4C33_8B9C_2C137F39E508_H__

#include "igris_c_can/can/can_typedef.h"

#include <bitset>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace robros {
namespace pdu_v2 {

constexpr uint8_t PDU_V2_CAN_ID = 0x60;

enum class pduv2ParsingError {
    NO_ERROR,
    HEADER_MISMATCH,
    DATA_NACK,
    NO_RELEVANT_ID,
    INVALID_LENGTH,
};

enum class pduv2Command : uint8_t {
    SAVE_BATTERY_ADC_LOW      = 0xA1,
    SAVE_BATTERY_ADC_HIGH     = 0xA2,
    SAVE_BATTERY_ADC_LEVEL_HL = 0x9E,
    RELAY_RESET               = 0xE0,
    GROUNDING_ENABLE          = 0xCB,
    ROBOT_STARTUP_SEQUENCE    = 0xEE,
    PRECHARGE_CHECK_ENABLE    = 0x07,
    PRECHARGE_THRESHOLD       = 0x17,
    PRECHARGE_TIME            = 0x49,
    LOW_VOLTAGE_WARNING_LEVEL = 0x36,
    PRODUCT_VERSION           = 0xFA,
};

constexpr uint8_t STATE_ESTOP_BIT            = 0b10000000;
constexpr uint8_t STATE_PRECHARGE_FINISH_BIT = 0b01000000;
constexpr uint8_t STATE_ACTION_FINISH_BIT    = 0b00100000;
constexpr uint8_t STATE_PRECHARGE_RELAY_BIT  = 0b00000100;
constexpr uint8_t STATE_HIGH_RELAY_BIT       = 0b00000010;
constexpr uint8_t STATE_LOW_RELAY_BIT        = 0b00000001;

constexpr uint16_t ERROR_NORMAL              = 0x0000;
constexpr uint16_t ERROR_ESTOP_PRESSED       = 0x0001;
constexpr uint16_t ERROR_LOW_BATT            = 0x0002;
constexpr uint16_t ERROR_HIGH_BATT           = 0x0004;
constexpr uint16_t ERROR_PRECHARGE_FAIL      = 0x0008;
constexpr uint16_t ERROR_POWER_SHORT_CIRCUIT = 0x0010;
constexpr uint16_t ERROR_TEMP1               = 0x0020;
constexpr uint16_t ERROR_TEMP2               = 0x0040;

// NTC raw value sentinel that indicates the firmware is emitting a fixed
// placeholder reading rather than a real sensor measurement (per protocol spec).
constexpr uint8_t NTC_FIXED_OUTPUT_SENTINEL = 0xCC;

// NTC raw is an 8-bit scaling of the 0~80 degC range.
constexpr float NTC_TEMP_FULL_SCALE_C = 80.0f;

// Battery level: DATA[4]=H, DATA[5]=L form a 16-bit raw value that the PDU
// firmware computes as (batt_rate% * 65535 / 100), where batt_rate is the
// position of the filtered ADC reading inside its [low_dec, high_dec]
// calibration window. The displayed volts use the firmware's curve
// batt_lev = 48.6 + 0.06 * batt_rate, i.e. linear from 48.6V (raw=0) to
// 54.6V (raw=65535). See igris_c_pdu/ROBROS/PDU/pdu.h L947, L955-L956.
constexpr float BATT_LEVEL_DISPLAY_BASE_V = 48.6f;
constexpr float BATT_LEVEL_DISPLAY_SPAN_V = 6.0f;
constexpr uint16_t BATT_LEVEL_RAW_MAX     = 0xFFFF;

// Precharge threshold: battery voltage * (data[1] / 255.0) is compared against
// the live power-line voltage. Default 0xB3 -> 35.1V at nominal battery.
constexpr uint8_t PRECHARGE_THRESHOLD_DEFAULT = 0xB3;
// Precharge time: total duration is (3000 * data[1] / 255) ms. Default 0xFF -> 3000ms.
constexpr uint8_t PRECHARGE_TIME_DEFAULT        = 0xFF;
constexpr uint16_t PRECHARGE_TIME_FULL_SCALE_MS = 3000;
// Low voltage warning: threshold = 48.6 + 6.0 * (data[1] / 255). Default 0x13 -> ~49.05V.
constexpr uint8_t LOW_VOLTAGE_WARNING_LEVEL_DEFAULT = 0x13;
constexpr float LOW_VOLTAGE_WARNING_BASE_V          = 48.6f;
constexpr float LOW_VOLTAGE_WARNING_SPAN_V          = 6.0f;

struct pduv2Status {
    uint64_t productVersion;

    uint8_t targetActionData;
    uint8_t stateVars;
    bool estopEnabled;
    bool prechargeFinished;
    bool actionFinished;
    bool prechargeRelayState;
    bool highRelayState;
    bool lowRelayState;

    uint16_t errorCode;

    uint16_t battLevelRaw;
    float battLevelV;
    uint8_t ntc1Raw;
    uint8_t ntc2Raw;
    bool ntc1FixedOutput;
    bool ntc2FixedOutput;
    float ntc1TempC;
    float ntc2TempC;

    uint32_t packetCount;
    bool connected;

    std::string toString() const {
        std::ostringstream oss;
        oss << "Product Version: " << productVersion << "\n"
            << "Target Action: 0x" << std::hex << static_cast<int>(targetActionData) << std::dec << "\n"
            << "State: 0b" << std::bitset<8>(stateVars) << "\n"
            << "  E-Stop Enabled: " << (estopEnabled ? "Yes" : "No") << "\n"
            << "  Precharge Finished: " << (prechargeFinished ? "Yes" : "No") << "\n"
            << "  Action Finished: " << (actionFinished ? "Yes" : "No") << "\n"
            << "  Precharge Relay: " << (prechargeRelayState ? "On" : "Off") << "\n"
            << "  HIGH Relay: " << (highRelayState ? "On" : "Off") << "\n"
            << "  LOW Relay: " << (lowRelayState ? "On" : "Off") << "\n"
            << "Error Code: 0x" << std::hex << errorCode << std::dec << "\n"
            << "Battery Level (raw 16b): 0x" << std::hex << battLevelRaw << std::dec << " (" << battLevelV << " V)\n"
            << "NTC1: raw=" << static_cast<int>(ntc1Raw) << ", temp=" << ntc1TempC << " C" << (ntc1FixedOutput ? " (FIXED)" : "") << "\n"
            << "NTC2: raw=" << static_cast<int>(ntc2Raw) << ", temp=" << ntc2TempC << " C" << (ntc2FixedOutput ? " (FIXED)" : "") << "\n"
            << "Packet Count: " << packetCount << "\n";
        return oss.str();
    }
};

namespace protocol {

class PacketHandler {
  public:
    virtual ~PacketHandler() = default;

    int registerPDUStatus(pduv2Status *status) {
        if (status == nullptr) {
            std::cerr << "[ERROR] pduv2Status pointer is null." << std::endl;
            return -1;
        }
        this->status_ = status;
        return 0;
    }

    pduv2Status *getPDUStatus() {
        if (status_ == nullptr) {
            std::cerr << "[ERROR] pduv2Status pointer is null." << std::endl;
            return nullptr;
        }
        return status_;
    }

  private:
    pduv2Status *status_ = nullptr;
};

template <class CanMessage> class ProtocolPDUv2 : public PacketHandler {
  public:
    ProtocolPDUv2()  = default;
    ~ProtocolPDUv2() = default;

    // Single-byte commands (DATA[0] only, LEN=1).
    static CanMessage getSaveBatteryAdcLowCommand();
    static CanMessage getSaveBatteryAdcHighCommand();
    static CanMessage getSaveBatteryAdcLevelHLCommand();
    static CanMessage getRelayResetCommand();
    static CanMessage getGroundingEnableCommand();
    static CanMessage getRobotStartupSequenceCommand();

    // Two-byte commands (DATA[0] = command, DATA[1] = value, LEN=2).
    static CanMessage getPrechargeCheckEnableCommand(bool enable);
    static CanMessage getPrechargeThresholdCommand(uint8_t value);
    static CanMessage getPrechargeTimeCommand(uint8_t value);
    static CanMessage getLowVoltageWarningLevelCommand(uint8_t value);

    // RTR query: requests an unsolicited Return Data Frame.
    static CanMessage getStatusQueryCommand();

    // Legacy command kept for backwards compatibility with existing apps.
    static CanMessage getProductVersionCommand();

    static pduv2ParsingError parseFeedbackMessage(const CanMessage &msg, pduv2Status &status);
};

}  // namespace protocol
}  // namespace pdu_v2
}  // namespace robros

#endif /* __PROTOCOL_PACKET_PDU_V2_AC98289C_AC63_4C33_8B9C_2C137F39E508_H__ */
