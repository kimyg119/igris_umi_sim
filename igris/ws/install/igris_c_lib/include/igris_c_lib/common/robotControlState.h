#ifndef __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_COMMON_ROBOTCONTROLSTATE_H__
#define __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_COMMON_ROBOTCONTROLSTATE_H__

#include <cstdint>

namespace robros::lib::common {

class RobotControlState {
  public:
    enum Value : uint32_t {
        NOT_READY = 0,
        IDLE,
        STOP,
        LOW,
        HIGH_MOTION_ACTIVE,
        HIGH_MOTION_STANDBY,
        WALK_LOW,
        WALK_HIGH_MOTION_ACTIVE,
        WALK_HIGH_MOTION_STANDBY,
        WHOLE_BODY_CONTROL,
    };

    RobotControlState() = default;
    constexpr RobotControlState(Value v) : value_(v) {}
    constexpr operator Value() const { return value_; }
    explicit operator bool() = delete;

    constexpr bool operator==(RobotControlState other) const { return value_ == other.value_; }
    constexpr bool operator!=(RobotControlState other) const { return value_ != other.value_; }

    bool isMotionActive() const { return value_ == HIGH_MOTION_ACTIVE || value_ == WALK_HIGH_MOTION_ACTIVE; }

    const char *toString() const {
        switch (value_) {
        case NOT_READY:
            return "NOT_READY";
        case IDLE:
            return "IDLE";
        case STOP:
            return "STOP";
        case LOW:
            return "LOW";
        case HIGH_MOTION_ACTIVE:
            return "HIGH_MOTION_ACTIVE";
        case HIGH_MOTION_STANDBY:
            return "HIGH_MOTION_STANDBY";
        case WALK_LOW:
            return "WALK_LOW";
        case WALK_HIGH_MOTION_ACTIVE:
            return "WALK_HIGH_MOTION_ACTIVE";
        case WALK_HIGH_MOTION_STANDBY:
            return "WALK_HIGH_MOTION_STANDBY";
        case WHOLE_BODY_CONTROL:
            return "WHOLE_BODY_CONTROL";
        default:
            return "UNKNOWN";
        }
    }

  private:
    Value value_ = NOT_READY;
};

}  // namespace robros::lib::common

#endif /* __IGRIS_C_LIB_INCLUDE_IGRIS_C_LIB_COMMON_ROBOTCONTROLSTATE_H__ */
