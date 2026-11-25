/**
 * @file end_effector.hpp
 * @author colson (colson@robros.co.kr)
 * @brief Per-type end-effector behaviour over an injected MotorBusInterface.
 *
 * EndEffectorInterface is the common contract (mirrors igris_c_controller's ControllerInterface /
 * BaseModule pattern). Concrete types (hand / 1dof / magnet) live in end_effector.cpp and are
 * created via makeEndEffector(); each owns the bus it was given, so the transport (Dynamixel today)
 * can be swapped without touching effector logic.
 * @version 0.1
 * @date 2026-06-18
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_END_EFFECTOR_HPP__
#define __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_END_EFFECTOR_HPP__

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "igris_c_hand/motor_bus_interface.hpp"

namespace robros::hand::v2 {

enum class EndEffectorMode : int {
    HAND    = 0,
    ONE_DOF = 1,  // single-DOF gripper (config token "1dof")
    MAGNET  = 2,
};

struct MotorState {
    int id                     = -1;
    bool alive                 = false;
    int32_t position           = 0;    // raw Dynamixel unit
    double normalized_position = 0.0;  // 0.0 to 1.0 based on extension/flexion refs
    double velocity            = 0.0;  // rad/s
    double current             = 0.0;  // mA
    int8_t temperature         = 0;    // Celsius
    uint8_t error_status       = 0;    // hardware error flags
};

using TargetsArray = std::array<double, 12>;

class EndEffectorInterface {
  public:
    virtual ~EndEffectorInterface() = default;

    // Type-specific calibration / homing. Throws std::runtime_error if the link is not open.
    virtual void initialize() = 0;
    // Apply normalized [0,1] targets (indexed by joint slot) for the given ids; others hold.
    virtual void writeTargets(const std::vector<double> &positions, const std::vector<int> &ids) = 0;
    // Read state for all 12 joint slots (indexed by joint slot, not id).
    virtual std::array<MotorState, 12> readState() = 0;

    // Reopen the underlying bus (close + open + verify) WITHOUT calibrating: torque stays off and a
    // manual initialize() is required afterwards. Used after a USB reconnect. Returns link-up.
    virtual bool reconnect() = 0;
    // Drop the underlying bus (close the port) so isOpen() reports false. Called on a detected USB
    // unplug: the bus does not close itself on a live disconnect, so without this the stale fd would
    // keep isOpen() true and block both reconnect() and a clean initialize() guard.
    virtual void close()        = 0;
    virtual bool isOpen() const = 0;  // lock-free
};

// Composition root: build the transport (DxlBus) for the mode and wrap it in the matching effector.
// The returned effector owns the bus. Returns nullptr for an unsupported mode.
std::unique_ptr<EndEffectorInterface> makeEndEffector(EndEffectorMode mode, const std::string &device_name);

}  // namespace robros::hand::v2

#endif  // __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_END_EFFECTOR_HPP__
