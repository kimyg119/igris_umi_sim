/**
 * @file motor_interface.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-08-07
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __SRC_IGRIS_C_CAN_INCLUDE_MOTOR_PACKET_HANDLER_H__
#define __SRC_IGRIS_C_CAN_INCLUDE_MOTOR_PACKET_HANDLER_H__

#include "igris_c_can/motor/motorTypeDef.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

namespace robros {
namespace motor {
namespace protocol {
class PacketHandler {
  public:
    PacketHandler() {}
    ~PacketHandler() = default;
};
}  // namespace protocol
}  // namespace motor
}  // namespace robros

#endif /* __SRC_IGRIS_C_CAN_INCLUDE_MOTOR_PACKET_HANDLER_H__ */
