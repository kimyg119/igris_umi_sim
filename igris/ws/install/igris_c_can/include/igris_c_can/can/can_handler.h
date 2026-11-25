#ifndef __IGRIS_C_CAN_INCLUDE_IGRIS_C_CAN_CAN_CAN_HANDLER_H__
#define __IGRIS_C_CAN_INCLUDE_IGRIS_C_CAN_CAN_CAN_HANDLER_H__

#include "igris_c_can/util/logger.h"

#include <cstdint>
#include <functional>
#include <iostream>

namespace robros {
namespace can {
template <class Hardware> class CANHandler {
  public:
    CANHandler() { logger_ = robros::util::Logger::get(); }
    ~CANHandler() = default;

  private:
    std::shared_ptr<spdlog::logger> logger_;
    Hardware _hardware;
    void onMessageReceived(int channel, void *msg) {
        robros::util::Logger::log(*logger_, spdlog::level::info, "Message received on channel " + std::to_string(channel));
    }

  public:
    Hardware *getHardware() { return &_hardware; }

    void initialize() {
        robros::util::Logger::log(*logger_, spdlog::level::info, "Initializing...");
        _hardware.initialize();
        robros::util::Logger::log(*logger_, spdlog::level::info, "Initialized.");
    }

    uint64_t run(int channel = -1) { return _hardware.spinOnce(channel); }
};

}  // namespace can
}  // namespace robros

// template class robros::can::CANHandler<robros::can::PCANController>;
// template class robros::can::CANHandler<robros::can::DummyCANController>;
// template class robros::can::CANHandler<robros::can::PCANController>;

#endif /* __IGRIS_C_CAN_INCLUDE_IGRIS_C_CAN_CAN_CAN_HANDLER_H__ */
