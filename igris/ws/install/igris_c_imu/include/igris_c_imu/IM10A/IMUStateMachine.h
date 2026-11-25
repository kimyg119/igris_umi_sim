/**
 * @file IMUStateMachine.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-12-31
 * @note
 * IMU State Machine for parsing incoming IMU data frames.
 * Protocol is described in the datasheet url below:
 * https://github.com/YahboomTechnology/10-axis_IMU_Module/blob/main/5.Communication%20protocol/10-axis%20IMU%20module%20communication%20protocol.pdf
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __IGRIS_C_IGRIS_C_IMU_INCLUDE_IGRIS_C_IMU_IMUSTATEMACHINE_H__
#define __IGRIS_C_IGRIS_C_IMU_INCLUDE_IGRIS_C_IMU_IMUSTATEMACHINE_H__

#include <iostream>
#include <sstream>
#include <stdint.h>

// ros
#include "igris_c_lib/util/time.h"
#include "igris_c_shm/logger/logger.h"

namespace robros::imu {
static constexpr uint16_t FRAME_LEN   = 11;
static constexpr uint8_t FRAME_HEADER = 0x55;
static constexpr float G_TO_MS2       = 9.80665f;
static constexpr float DEG_TO_RAD     = (3.14159265358979323846f / 180.0f);

enum class FrameState { WAIT_FOR_HEADER, WAIT_FOR_TYPE, COLLECT_DATA, VALIDATE_CHECKSUM };

enum class FrameType {
    FRAME_TYPE_ACC     = 0x51,
    FRAME_TYPE_GYRO    = 0x52,
    FRAME_TYPE_RPY     = 0x53,
    FRAME_TYPE_QUAT    = 0x59,
    FRAME_TYPE_INVALID = 0x00
};

struct IMUData {
    float acc[3];
    float gyro[3];
    float rpy[3];
    float quat[4];

    uint64_t accCount;
    uint64_t gyroCount;
    uint64_t rpyCount;
    uint64_t quatCount;

    uint64_t now_acc_time;
    uint64_t last_acc_time;
    uint64_t diff_acc_time;

    uint64_t now_gyro_time;
    uint64_t last_gyro_time;
    uint64_t diff_gyro_time;

    uint64_t now_rpy_time;
    uint64_t last_rpy_time;
    uint64_t diff_rpy_time;

    uint64_t now_quat_time;
    uint64_t last_quat_time;
    uint64_t diff_quat_time;

    IMUData()
        : acc{0.0f}, gyro{0.0f}, rpy{0.0f}, quat{0.0f}, accCount(0), gyroCount(0), rpyCount(0), quatCount(0), now_acc_time(0),
          last_acc_time(0), diff_acc_time(0), now_gyro_time(0), last_gyro_time(0), diff_gyro_time(0), now_rpy_time(0), last_rpy_time(0),
          diff_rpy_time(0), now_quat_time(0), last_quat_time(0), diff_quat_time(0) {}

    void clearCount() {
        accCount  = 0;
        gyroCount = 0;
        rpyCount  = 0;
        quatCount = 0;
    }

    std::string toString() {
        std::ostringstream oss;
        oss << "Acc (m/s^2): [" << acc[0] << ", " << acc[1] << ", " << acc[2] << "] "
            << "diff_acc_time (ms): " << diff_acc_time / 1000000.0f << "\n"
            << "Gyro (rad/s): [" << gyro[0] << ", " << gyro[1] << ", " << gyro[2] << "] "
            << "diff_gyro_time(ms): " << diff_gyro_time / 1000000.0f << "\n"
            << "RPY (deg): [" << rpy[0] << ", " << rpy[1] << ", " << rpy[2] << "]" << " "
            << "diff_rpy_time(ms): " << diff_rpy_time / 1000000.0f << "\n"
            << "Quat: [" << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "] "
            << "diff_quat_time(ms): " << diff_quat_time / 1000000.0f << "\n";
        return oss.str();
    }
};

class IMUStateMachine {
  public:
  private:
    FrameState state;
    uint8_t buffer[FRAME_LEN];
    int buffer_index;
    uint8_t checksum;

  public:
    IMUStateMachine() : state(FrameState::WAIT_FOR_HEADER), buffer_index(0), checksum(0) {
        for (int i = 0; i < FRAME_LEN; i++) {
            buffer[i] = 0;
        }
    }

    bool process_byte(uint8_t byte, IMUData &imu_data) {

        switch (state) {
        case FrameState::WAIT_FOR_HEADER:
            if (byte == robros::imu::FRAME_HEADER) {
                buffer[0]    = byte;
                buffer_index = 1;
                state        = FrameState::WAIT_FOR_TYPE;
            }
            break;

        case FrameState::WAIT_FOR_TYPE:
            buffer[buffer_index++] = byte;
            if (byte == static_cast<uint8_t>(FrameType::FRAME_TYPE_ACC) || byte == static_cast<uint8_t>(FrameType::FRAME_TYPE_GYRO) ||
                byte == static_cast<uint8_t>(FrameType::FRAME_TYPE_RPY) || byte == static_cast<uint8_t>(FrameType::FRAME_TYPE_QUAT)) {
                state = FrameState::COLLECT_DATA;
            } else {
                robros::shm::logger::get_logger("igris_c_imu").error("Invalid frame type: 0x" + std::to_string(static_cast<int>(byte)));
                reset();
            }
            break;

        case FrameState::COLLECT_DATA:
            buffer[buffer_index++] = byte;
            if (buffer_index < FRAME_LEN - 1) {
                // collecting data bytes
            } else {
                state = FrameState::VALIDATE_CHECKSUM;
            }
            break;

        case FrameState::VALIDATE_CHECKSUM:
            buffer[buffer_index]     = byte;
            uint64_t current_time_ns = robros::util::time::get_timestamp_ns();
            checksum                 = 0;
            for (int i = 0; i < 10; i++)
                checksum += buffer[i];

            if (checksum == byte) {
                int16_t raw_data[4];
                memcpy(raw_data, buffer + 2, 8);
                // Convert raw data to actual values (scale factor: 32768/16g for acc, 32768/2000dps for gyro, 32768/180deg for rpy)
                float accScale  = 16.0f / 32768.0f;
                float gyroScale = 2000.0f / 32768.0f;
                float rpyScale  = 180.0f / 32768.0f;
                float quatScale = 1.0f / 32768.0f;

                if (buffer[1] == static_cast<uint8_t>(FrameType::FRAME_TYPE_ACC)) {
                    for (int i = 0; i < 3; i++) {
                        imu_data.acc[i] = raw_data[i] * accScale * robros::imu::G_TO_MS2;
                    }
                    imu_data.accCount++;
                    imu_data.now_acc_time  = current_time_ns;
                    imu_data.diff_acc_time = imu_data.now_acc_time - imu_data.last_acc_time;
                    imu_data.last_acc_time = current_time_ns;
                } else if (buffer[1] == static_cast<uint8_t>(FrameType::FRAME_TYPE_GYRO)) {
                    for (int i = 0; i < 3; i++) {
                        imu_data.gyro[i] = raw_data[i] * gyroScale * robros::imu::DEG_TO_RAD;
                    }
                    imu_data.gyroCount++;
                    imu_data.now_gyro_time  = current_time_ns;
                    imu_data.diff_gyro_time = imu_data.now_gyro_time - imu_data.last_gyro_time;
                    imu_data.last_gyro_time = current_time_ns;
                } else if (buffer[1] == static_cast<uint8_t>(FrameType::FRAME_TYPE_RPY)) {
                    for (int i = 0; i < 3; i++) {
                        imu_data.rpy[i] = raw_data[i] * rpyScale;
                    }
                    imu_data.rpyCount++;
                    imu_data.now_rpy_time  = current_time_ns;
                    imu_data.diff_rpy_time = imu_data.now_rpy_time - imu_data.last_rpy_time;
                    imu_data.last_rpy_time = current_time_ns;
                } else if (buffer[1] == static_cast<uint8_t>(FrameType::FRAME_TYPE_QUAT)) {
                    for (int i = 0; i < 4; i++) {
                        imu_data.quat[i] = raw_data[i] * quatScale;
                    }
                    imu_data.quatCount++;
                    imu_data.now_quat_time  = current_time_ns;
                    imu_data.diff_quat_time = imu_data.now_quat_time - imu_data.last_quat_time;
                    imu_data.last_quat_time = current_time_ns;
                } else {
                }

                state        = FrameState::WAIT_FOR_HEADER;
                buffer_index = 0;
                reset();
                return true;  // Frame complete and valid
            } else {
                std::stringstream ss;

                ss << "Buffer: ";
                for (int i = 0; i < FRAME_LEN; i++) {
                    ss << std::hex << static_cast<int>(buffer[i]) << " ";
                }
                ss << std::dec << std::endl;
                robros::shm::logger::get_logger("igris_c_imu").debug(ss.str());
                robros::shm::logger::get_logger("igris_c_imu")
                    .error("Checksum error: expected 0x" + std::to_string(checksum) + ", got 0x" + std::to_string(byte));
                reset();
            }
            break;
        }
        return false;
    }

    void reset() {
        state        = FrameState::WAIT_FOR_HEADER;
        buffer_index = 0;
        checksum     = 0;
    }
};

}  // namespace robros::imu

#endif /* __IGRIS_C_IGRIS_C_IMU_INCLUDE_IGRIS_C_IMU_IMUSTATEMACHINE_H__ */
