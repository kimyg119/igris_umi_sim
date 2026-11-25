#ifndef __LED_FA7CB8B0_53A6_465A_B48B_3321A50AA178_H__
#define __LED_FA7CB8B0_53A6_465A_B48B_3321A50AA178_H__

#include "igris_c_can/pcan/PCANBasic.h"

#include <bitset>
#include <cstdint>
#include <iostream>
#include <sstream>

namespace robros {
namespace led {

namespace protocol {

class PacketHandler {
  public:
    virtual ~PacketHandler() = default;
};

template <class CanMessage> class ProtocolLed : public PacketHandler {
  public:
    ProtocolLed()  = default;
    ~ProtocolLed() = default;

    enum class LedPosition : uint16_t {
        HEAD          = 0x710,
        LEFT_ARM      = 0x711,
        RIGHT_ARM     = 0x712,
        LEFT_LEG      = 0x713,
        RIGHT_LEG     = 0x714,
        ALL_POSITIONS = 0x730,
    };

    enum class Circumstance {
        Booting,
        BootDone,
        BatteryOver30Percent,
        BatteryUnder30Percent,
        BatteryUnder20Percent,
        BatteryUnder10Percent,
        Error,
        EStop,
        ReleaseEStopAndOk,
        ReleaseEStopAndError,
        PowerOffStandby,
    };

    enum class LedCommand : uint8_t {
        HeadControl     = 0x00,
        SetWeight       = 0x01,
        ReadWeight      = 0x02,
        ChangeId        = 0x03,
        LedOff          = 0x04,
        LedOn           = 0x05,
        QueryExistance  = 0x06,
        SetLedLowValue  = 0x07,
        SetLedHighValue = 0x08,
        SetTorqueRange  = 0x09,
        GetStatus1      = 0x0A,
        GetStatus2      = 0x0B,
        GetStatus3      = 0x0C,
    };

    static CanMessage bootLed();
    static CanMessage bootDoneLed();
    static CanMessage BatteryOver30Led();
    static CanMessage BatteryUnder30Led();
    static CanMessage BatteryUnder20Led();
    static CanMessage BatteryUnder10Led();
    static CanMessage ErrorLed();
    static CanMessage eStopLed();
    static CanMessage PowerOffStandby();
    static CanMessage resetLed();

    // Maps a high-level robot state to the corresponding head LED command.
    static CanMessage genMessageByCircumstance(Circumstance circumstance);

    static CanMessage ChangeLedDriverIdCommand(LedPosition position, LedPosition newPosition);
    static CanMessage SetLedLowValueCommand(LedPosition position, uint8_t lowValue[3]);
    static CanMessage SetLedHighValueCommand(LedPosition position, uint8_t HighValue[3]);
    static CanMessage SetCurrentRangeCommand(LedPosition position, uint8_t minCurrent, uint8_t maxCurrent);

    // Brightness weights are clamped to the device-supported range internally.
    static CanMessage setLedWeightCommand(LedPosition position, uint8_t slot1Weight, uint8_t slot2Weight);

    static CanMessage readLedWeightCommand(LedPosition position);
    static CanMessage setLedOffCommand(LedPosition position);
    static CanMessage setLedOnCommand(LedPosition position);
    static CanMessage queryLedExistenceCommand(LedPosition position);
    static CanMessage setLedLowValueCommand(LedPosition position, uint8_t slot1LowValue[3], uint8_t slot2LowValue[3]);
    static CanMessage setLedHighValueCommand(LedPosition position, uint8_t slot1HighValue[3], uint8_t slot2HighValue[3]);
    static CanMessage getDriverStatusCommand1(LedPosition position);
    static CanMessage getDriverStatusCommand2(LedPosition position);
    static CanMessage getDriverStatusCommand3(LedPosition position);
};
}  // namespace protocol
}  // namespace led
}  // namespace robros

#endif /* __LED_FA7CB8B0_53A6_465A_B48B_3321A50AA178_H__ */
