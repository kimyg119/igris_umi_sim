/**
 * @file igris_c.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-09-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __IGRIS_C_CB3EF12E_70FD_4946_97C3_32853ACDF287_H__
#define __IGRIS_C_CB3EF12E_70FD_4946_97C3_32853ACDF287_H__

#include "igris_c_can/can/can_impl.h"
#include "igris_c_can/motor/motorTypeDef.h"
#include "igris_c_can/motor/protocol_packet_bionic.h"
#include "igris_c_can/motor/protocol_packet_v4.h"
#include "igris_c_shm/shm/types.h"
#include "yaml-cpp/yaml.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <thread>

namespace robros::controller::igris_c {
class JointManager;
}

namespace robros::lib::util {
class RtSingleTask;
}

namespace robros::can {

enum class PduDisconnectBehavior { Stop, Running };

template <class CanHandler> class IGRIS_C {
  public:
    IGRIS_C(const std::vector<std::string> &motorConfigPaths, const std::string &parallelLinksPath, uint8_t PDULowAdcValue,
            uint8_t PDUHighAdcValue, const std::string &disconnectBehavior, bool setupEnabled = true);
    ~IGRIS_C();
    IGRIS_C(const IGRIS_C &)            = delete;
    IGRIS_C &operator=(const IGRIS_C &) = delete;

    int updateConfigStatus(std::string configPath);

    int initializeHardwareAll();

    int offsetCapture();

    void CANReceiveCallback(int channel, TPCANMsg *msg);

    int checkPDUVersion();
    int powerOn();
    int powerOff();
    int torqueOn();
    int torqueOff();

    std::vector<std::string> run();

  public:
    std::vector<robros::motor::motorConfig> getMotorConfigs();
    std::vector<robros::motor::motorStatus> getMotorStatuses();

  private:
    void sendMessage(uint16_t channel, TPCANMsg *msg);
    bool sendRecvMessage(uint16_t channel, TPCANMsg *sendMsg, uint64_t microTimeout, TPCANMsg *recvMsg, uint16_t expectedId);
    void delayUsWithCANReceive(uint64_t microseconds, int channel);

    void fetchCommand();
    void pushState();

    void control(int channel);
    void zeroTorque(int channel);
    void readPosition(int channel);
    void errorCheck(int ch, uint16_t id);

    void workingChannel(int channel);

  private:
    std::vector<std::string> motorConfigPaths_;
    std::shared_ptr<CanHandler> canHandler_;
    std::mutex CANMutex_;
    std::mutex channelMutex_[4];

    bool isRunning_;
    bool isInitialized_;
    bool isTorqueOn_;
    bool isPowerOn_;
    bool nowInitialize;

    // atomic so the per-service worker threads can compare-exchange it as a
    // mutual-exclusion gate (reject a service request while another is running).
    std::atomic<bool> nowService_;

    std::thread commandFetchThread_;
    std::thread channelThreads_[4];

    motor::MotorControlProtocolVersion motorProtocolVersion_;
    robros::shm::shm::canShm_st *shmPtr = nullptr;
    std::size_t jointCmdConsumerId_     = 0;
    bool jointCmdRegistered_            = false;

  public:
    int execute();

    // The controller-owned CAN segment (created in the constructor). Exposed so the CAN node can
    // attach the embedded On/Off service servers (bms/torque/initialize) after construction.
    robros::shm::shm::canShm_st *canShm() const { return shmPtr; }

  private:
    robros::lib::util::RtSingleTask *commandFetchTask_;
    robros::lib::util::RtSingleTask *pushStateTask_;
    robros::lib::util::RtSingleTask *channelControlTasks_[4];
    uint8_t motorChannelIndexCount_[4] = {0, 0, 0, 0};
    bool isTaskRun_;
    bool stopTask_;

    uint8_t PDULowValue;
    uint8_t PDUHighValue;

    // PDU CAN-disconnect control policy (see igris_c.can.pdu.disconnect_behavior).
    PduDisconnectBehavior pduDisconnectBehavior_ = PduDisconnectBehavior::Stop;

    // Set true whenever a service pauses the PDU status queries (initializeHardwareAll
    // stops the channel/pushState tasks; other services hold all channel mutexes).
    // Consumed by pushState() to open a fresh PDU-liveness window on resume, so the
    // query gap during a service is not misread as a PDU disconnect.
    std::atomic<bool> pduLivenessResync_{false};

    // Motor setup policy (see igris_c.can.setup). When false, initializeHardwareAll() skips all
    // motor write/setup commands and only verifies motor presence via read-back.
    bool setupEnabled_ = true;

  private:
    std::unique_ptr<robros::controller::igris_c::JointManager> jointManager_;
};
}  // namespace robros::can

#endif /* __IGRIS_C_CB3EF12E_70FD_4946_97C3_32853ACDF287_H__ */
