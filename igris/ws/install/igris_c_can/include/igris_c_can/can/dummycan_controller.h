/**
 * @file dummycan_controller.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-08-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __IGRIS_C_CAN_INCLUDE_CAN_DUMMYCAN_CONTROLLER_H__
#define __IGRIS_C_CAN_INCLUDE_CAN_DUMMYCAN_CONTROLLER_H__

#include "can_interface.h"

#include <cstdint>
#include <iostream>

namespace robros {
namespace can {
class DummyCANController : public CANInterface {
  public:
    DummyCANController() = default;
    DummyCANController(int argc, char *argv[]) { std::cout << "Dummy CAN Controller initialized with arguments." << std::endl; }
    ~DummyCANController() { std::cout << "Dummy CAN Controller destroyed." << std::endl; }

    void getMessage(int channel, void *msg) override {
        std::cout << "Dummy CAN Controller: Getting message from channel " << channel << std::endl;
    }

    void sendMessage(int channel, const void *msg) override {
        std::cout << "Dummy CAN Controller: Sending message on channel " << channel << std::endl;
    }

    void initialize() override { std::cout << "Dummy CAN Controller initialized." << std::endl; }

    uint64_t spinOnce(int channel) override {
        std::cout << "Dummy CAN Controller spinning on channel " << channel << std::endl;
        return 0;
    }
};
}  // namespace can
}  // namespace robros

#endif /* __IGRIS_C_CAN_INCLUDE_CAN_DUMMYCAN_CONTROLLER_H__ */
