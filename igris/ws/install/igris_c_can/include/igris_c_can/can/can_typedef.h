#ifndef __SRC_IGRIS_C_CAN_INCLUDE_CAN_CAN_TYPEDEF_H__
#define __SRC_IGRIS_C_CAN_INCLUDE_CAN_CAN_TYPEDEF_H__

#include "igris_c_can/pcan/PCANBasic.h"

#include <cstdint>
#include <vector>

namespace robros {
namespace can {

class CANMessageBase {
  public:
    CANMessageBase(uint32_t id, std::vector<uint8_t> data, uint8_t msgType) : id_(id), data_{data}, msgType_(msgType) {}

  protected:
    uint32_t id_;
    std::vector<uint8_t> data_;
    uint8_t msgType_;
};
}  // namespace can
}  // namespace robros

namespace robros {
namespace can {
class TPCANMessage : public CANMessageBase {
    typedef TPCANMsg message;

  public:
    TPCANMessage(uint32_t id, std::vector<uint8_t> data, uint8_t msgType) : CANMessageBase(id, data, msgType) {}
    TPCANMessage(message msg) : CANMessageBase(msg.ID, std::vector<uint8_t>(msg.DATA, msg.DATA + msg.LEN), msg.MSGTYPE) {}

    static TPCANMessage genMessage(uint32_t id, std::vector<uint8_t> data, uint8_t msgType) {
        TPCANMessage msg(id, data, msgType);
        return msg;
    }

    message &getMessage() {
        msg_.ID  = id_;
        msg_.LEN = data_.size();
        for (int i = 0; i < data_.size(); ++i) {
            msg_.DATA[i] = data_[i];
        }
        msg_.MSGTYPE = msgType_;
        return msg_;
    }

    operator message() { return getMessage(); }

    uint32_t getId() const { return id_; }

    std::vector<uint8_t> getData() const { return data_; }

  private:
    message msg_;
};
}  // namespace can
}  // namespace robros

#endif /* __SRC_IGRIS_C_CAN_INCLUDE_CAN_CAN_TYPEDEF_H__ */
