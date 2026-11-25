/**
 * @file dxl_bus.hpp
 * @author colson (colson@robros.co.kr)
 * @brief Dynamixel-over-serial implementation of MotorBusInterface.
 *
 * Owns the serial port + Dynamixel protocol handlers and exposes the bus primitives the
 * EndEffector classes use. Pure transport/protocol: it knows the configured motor ids + baud but
 * has no end-effector (hand/1dof/magnet) semantics. Not internally locked (see MotorBusInterface).
 * @version 0.1
 * @date 2026-06-18
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_DXL_BUS_HPP__
#define __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_DXL_BUS_HPP__

#include <atomic>
#include <dynamixel_sdk/dynamixel_sdk.h>
#include <string>
#include <vector>

#include "igris_c_hand/motor_bus_interface.hpp"

namespace robros::hand::v2 {

// Control table addresses (EEPROM)
constexpr uint16_t ADDR_DRIVE_MODE        = 10;
constexpr uint16_t ADDR_OPERATING_MODE    = 11;
constexpr uint16_t ADDR_HOMING_OFFSET     = 20;
constexpr uint16_t ADDR_TEMPERATURE_LIMIT = 31;
constexpr uint16_t ADDR_CURRENT_LIMIT     = 38;
constexpr uint16_t ADDR_VELOCITY_LIMIT    = 44;

// Control table addresses (RAM)
constexpr uint16_t ADDR_TORQUE_ENABLE       = 64;
constexpr uint16_t ADDR_LED                 = 65;
constexpr uint16_t ADDR_HARDWARE_ERROR      = 70;
constexpr uint16_t ADDR_POSITION_D_GAIN     = 80;
constexpr uint16_t ADDR_POSITION_P_GAIN     = 84;
constexpr uint16_t ADDR_GOAL_CURRENT        = 102;
constexpr uint16_t ADDR_GOAL_POSITION       = 116;
constexpr uint16_t ADDR_PRESENT_CURRENT     = 126;
constexpr uint16_t ADDR_PRESENT_VELOCITY    = 128;
constexpr uint16_t ADDR_PRESENT_POSITION    = 132;
constexpr uint16_t ADDR_PRESENT_TEMPERATURE = 146;

constexpr uint8_t OM_CURRENT                = 0;
constexpr uint8_t OM_CURRENT_BASED_POSITION = 5;

constexpr double PROTOCOL_VERSION = 2.0;

class DxlBus : public MotorBusInterface {
  public:
    // ids = motors this bus verifies on open() (reboot/ping); baud = serial baud rate.
    DxlBus(const std::string &device_name, int baud_rate, std::vector<int> ids);
    ~DxlBus() override;

    DxlBus(const DxlBus &)            = delete;
    DxlBus &operator=(const DxlBus &) = delete;

    bool open() override;
    void close() override;
    bool isOpen() const override { return is_open_.load(std::memory_order_acquire); }

    bool ping(int id, uint8_t *error = nullptr) override;
    bool reboot(int id) override;
    int setTorque(int id, bool on) override;
    void setTorques(const std::vector<int> &ids, bool on) override;
    int setOperatingMode(int id, uint8_t mode) override;
    void setOperatingModes(const std::vector<int> &ids, uint8_t mode) override;
    int setCurrentLimit(int id, uint16_t limit) override;
    void setCurrentLimits(const std::vector<int> &ids, uint16_t limit) override;
    int setGoalCurrent(int id, int16_t current) override;

    int32_t readPosition(int id) override;
    uint8_t readError(int id) override;
    uint8_t readTemperature(int id) override;

    void syncWriteGoalPositions(const std::vector<int> &ids, const std::vector<int32_t> &positions) override;
    std::vector<MotorFeedback> syncReadFeedback(const std::vector<int> &ids) override;

  private:
    static void divideParamByte(std::vector<uint8_t> &param, int data, int byte_size);

    std::string device_name_;
    int baud_rate_ = 0;
    std::vector<int> ids_;
    dynamixel::PortHandler *port_handler_     = nullptr;
    dynamixel::PacketHandler *packet_handler_ = nullptr;
    std::atomic<bool> is_open_{false};
};

}  // namespace robros::hand::v2

#endif  // __IGRIS_C_HAND_INCLUDE_IGRIS_C_HAND_DXL_BUS_HPP__
