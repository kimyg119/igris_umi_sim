/**
 * @file motor_bus_interface.hpp
 * @author colson (colson@robros.co.kr)
 * @brief Transport/protocol abstraction for a bank of position/current controlled motors.
 *
 * DxlBus is the Dynamixel-over-serial implementation; an alternative hardware backend implements
 * the same surface so the EndEffector classes can be retargeted without change. Methods are NOT
 * internally serialised: the owning EndEffector serialises whole operations (initialize / readState
 * / writeTargets / reconnect) under one mutex, matching the single request/response nature of the
 * bus. isOpen() is the exception and must be lock-free (polled by the node from another thread).
 * @version 0.1
 * @date 2026-06-18
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_MOTOR_BUS_INTERFACE_HPP__
#define __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_MOTOR_BUS_INTERFACE_HPP__

#include <cstdint>
#include <vector>

namespace robros::hand::v2 {

// Raw (protocol-unit) feedback for one motor from a bulk read; valid == false on a comm failure.
struct MotorFeedback {
    int32_t position = 0;
    int32_t velocity = 0;
    int16_t current  = 0;
    bool valid       = false;
};

class MotorBusInterface {
  public:
    virtual ~MotorBusInterface() = default;

    // Open + verify the link to the configured motors (open port, reboot, ping). Returns true when
    // all configured motors respond; leaves the port closed on failure. close() releases the port.
    virtual bool open()         = 0;
    virtual void close()        = 0;
    virtual bool isOpen() const = 0;  // lock-free

    // Per-motor primitives (raw protocol units). The single-id writes return the transport result
    // code (COMM_SUCCESS on success) so callers that branch on success behave as before; the batch
    // variants log per-id failures internally. ping() optionally returns the device error byte.
    virtual bool ping(int id, uint8_t *error = nullptr)                        = 0;
    virtual bool reboot(int id)                                                = 0;
    virtual int setTorque(int id, bool on)                                     = 0;
    virtual void setTorques(const std::vector<int> &ids, bool on)              = 0;
    virtual int setOperatingMode(int id, uint8_t mode)                         = 0;
    virtual void setOperatingModes(const std::vector<int> &ids, uint8_t mode)  = 0;
    virtual int setCurrentLimit(int id, uint16_t limit)                        = 0;
    virtual void setCurrentLimits(const std::vector<int> &ids, uint16_t limit) = 0;
    virtual int setGoalCurrent(int id, int16_t current)                        = 0;

    virtual int32_t readPosition(int id)    = 0;  // raw position; 0 on comm/device error
    virtual uint8_t readError(int id)       = 0;
    virtual uint8_t readTemperature(int id) = 0;

    // Group write goal positions (parallel arrays). Bulk read present position/velocity/current for
    // ids (result parallel to ids; entries .valid == false on comm failure / not read).
    virtual void syncWriteGoalPositions(const std::vector<int> &ids, const std::vector<int32_t> &positions) = 0;
    virtual std::vector<MotorFeedback> syncReadFeedback(const std::vector<int> &ids)                        = 0;
};

}  // namespace robros::hand::v2

#endif  // __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_MOTOR_BUS_INTERFACE_HPP__
