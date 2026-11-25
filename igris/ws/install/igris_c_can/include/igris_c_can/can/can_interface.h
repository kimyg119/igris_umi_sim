/**
 * @file can_interface.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-08-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __IGRIS_C_CAN_INCLUDE_CAN_CAN_INTERFACE_H__
#define __IGRIS_C_CAN_INCLUDE_CAN_CAN_INTERFACE_H__

#include <cstdint>
#include <functional>

namespace robros {
namespace can {

class CANInterface {
  public:
    CANInterface()  = default;
    ~CANInterface() = default;

    virtual void initialize()                              = 0;
    virtual void getMessage(int channel, void *msg)        = 0;
    virtual void sendMessage(int channel, const void *msg) = 0;
    /**
     * @brief Spin once to process incoming messages and handle any necessary operations.
     */
    virtual uint64_t spinOnce(int channel) = 0;
};

}  // namespace can
}  // namespace robros

#endif /* __IGRIS_C_CAN_INCLUDE_CAN_CAN_INTERFACE_H__ */
