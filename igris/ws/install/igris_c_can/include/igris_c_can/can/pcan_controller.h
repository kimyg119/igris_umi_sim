/**
 * @file pcan_controller.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-08-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __SRC_IGRIS_C_CAN_INCLUDE_CAN_PCAN_CONTROLLER_H__
#define __SRC_IGRIS_C_CAN_INCLUDE_CAN_PCAN_CONTROLLER_H__

#include "can_interface.h"
#include "igris_c_can/pcan/PCANBasic.h"

#include <cstdint>

namespace robros {
namespace can {
class PCANController : public CANInterface {

  public:
    PCANController() = default;
    PCANController(int argc, char *argv[]);
    ~PCANController() = default;

    std::function<void(int channel, TPCANMsg *msg)> onMessageReceived;

    void initialize() override;
    void getMessage(int channel, void *msg) override;
    void sendMessage(int channel, const void *msg) override;
    uint64_t spinOnce(int channel = -1) override;

    uint32_t getError(int channel);
    void resetError(int channel);

    bool sendRecvMessage(int channel, const void *msg, uint64_t microTimeout, TPCANMsg *receiveMsg, uint16_t expectedId = 0x0000);
};
}  // namespace can
}  // namespace robros

#endif /* __SRC_IGRIS_C_CAN_INCLUDE_CAN_PCAN_CONTROLLER_H__ */
