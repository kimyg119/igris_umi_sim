#ifndef __PROTOCOL_PACKET_PDU_V1_AA699C19_89A5_4F62_A10B_6AB87F276B99_H__
#define __PROTOCOL_PACKET_PDU_V1_AA699C19_89A5_4F62_A10B_6AB87F276B99_H__

#include "igris_c_can/pcan/PCANBasic.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <sstream>

#define ACK_bit 0b10000000
#define ESTOP_STATE_bit 0b01000000
#define pre_charge_relay 0b00000100
#define LEG_relay 0b00000010
#define BODY_relay 0b00000001
#define LEG_volt 0b00100000
#define BODY_volt 0b00010000

namespace robros {
namespace pdu_v1 {

const uint8_t STATE_RETURN_ID   = 0x62;
const uint8_t COMMAND_RETURN_ID = 0x52;

enum pduv1ParsingError {
    NO_ERROR,
    HEADER_MISMATCH,
    DATA_NACK,
    NO_RELEVANT_ID,
};

struct pduv1Status {
    uint8_t state;
    bool isPreCharging;
    bool isOnBodyPower;
    bool isOnLegsPower;
    uint8_t rawBatLevel;
    float batLevel;
    float batPercentage;
    bool linePowerState1;
    bool linePowerState2;
    bool estopState;
    bool connected;
    uint32_t packetCount;

    std::string toString() const {
        std::ostringstream oss;
        oss << "State: 0b" << std::bitset<8>(state) << "\n"
            << "Is Pre-Charging: " << (isPreCharging ? "Yes" : "No") << "\n"
            << "Is On Body Power: " << (isOnBodyPower ? "Yes" : "No") << "\n"
            << "Is On Legs Power: " << (isOnLegsPower ? "Yes" : "No") << "\n"
            << "Raw Battery Level: " << static_cast<int>(rawBatLevel) << "\n"
            << "Battery Level: " << batLevel << "\n"
            << "Battery Percentage: " << batPercentage << "%\n"
            << "Line Power State 1: " << (linePowerState1 ? "Yes" : "No") << "\n"
            << "Line Power State 2: " << (linePowerState2 ? "Yes" : "No") << "\n"
            << "E-Stop State: " << (estopState ? "Yes" : "No") << "\n";
        return oss.str();
    }
};

namespace protocol {

class PacketHandler {
  public:
    virtual ~PacketHandler() = default;

    int registerBMSStatus(pduv1Status *PDUStatuses) {
        if (PDUStatuses == nullptr) {
            std::cerr << "[ERROR] PDUStatuses pointer is null." << std::endl;
            return -1;
        }
        this->PDUStatuses_ = PDUStatuses;
        return 0;
    }

    pduv1Status *getBMSStatus() {
        if (PDUStatuses_ == nullptr) {
            std::cerr << "[ERROR] PDUStatuses pointer is null." << std::endl;
            return nullptr;
        }
        return PDUStatuses_;
    }

  private:
    pduv1Status *PDUStatuses_ = nullptr;
};

template <class CanMessage> class ProtocolPDUv1 : public PacketHandler {
  public:
    ProtocolPDUv1()  = default;
    ~ProtocolPDUv1() = default;

    static CanMessage setPowerOffCommand();
    static CanMessage setPreChargingCommand();
    static CanMessage setLegPowerOnCommand();
    static CanMessage setBodyPowerOnCommand();
    static CanMessage setPreChargeAndAllPowerOnCommand();
    static CanMessage setAllPowerOnCommand();
    static CanMessage getStatusMessageCommand();
    static pduv1ParsingError parseFeedbackMessage(const CanMessage &msg, pduv1Status &status, uint8_t low_adc_value = 220,
                                                  uint8_t high_adc_value = 246);
};
}  // namespace protocol
}  // namespace pdu_v1
}  // namespace robros

#endif /* __PROTOCOL_PACKET_PDU_V1_AA699C19_89A5_4F62_A10B_6AB87F276B99_H__ */
