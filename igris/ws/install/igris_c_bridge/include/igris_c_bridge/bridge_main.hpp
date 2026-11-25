#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "igris_c_bridge/bridge_state.hpp"
#include "igris_c_bridge/data_store.hpp"
#include "igris_c_bridge/dds/dds_manager.hpp"
#include "igris_c_bridge/ros2/ros2_manager.hpp"
#include "igris_c_bridge/schema/command_schema.hpp"
#include "igris_c_bridge/shm/shm_command_writer.hpp"
#include "igris_c_bridge/tcp/tcp_server.hpp"
#include "igris_c_bridge/types.hpp"
#include "igris_c_shm/logger/logger.h"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge {

namespace shm {
class ShmSettingsService;
}

class BridgeMain {
  public:
    // env identifies which build is running so the bridge can publish the
    // correct RobotEnvironment in RobotState. sim.cpp passes ROBOT_ENV_SIM,
    // real.cpp passes ROBOT_ENV_REAL (default).
    BridgeMain(int argc, char *argv[], igris_c::msg::dds::RobotEnvironment env = igris_c::msg::dds::RobotEnvironment::ROBOT_ENV_REAL);

    // Common initialization: config, SPSC SHM, schema, DataStore, BridgeState,
    // command writer, service client, TCP server, DDS manager.
    // Returns 0 on success, non-zero on failure.
    int initialize();

    // Start the sensor reader, spin ROS2, wait for shutdown, then cleanup.
    // ReaderT must provide start() and stop() methods.
    template <typename ReaderT> int run(std::shared_ptr<ReaderT> sensor_reader);

    std::shared_ptr<DataStore> dataStore() const { return data_store_; }
    std::shared_ptr<BridgeState> state() const { return state_; }
    std::shared_ptr<dds::DdsManager> ddsManager() const { return dds_manager_; }
    std::shared_ptr<ros2::Ros2Manager> ros2Manager() const { return ros2_manager_; }
    robros::shm::shm::controlShm_st *spscPtr() const { return spsc_ptr_; }
    std::atomic<bool> &running() { return running_; }

    // The SHM settings service (CAN On/Off clients + hand init). Exposed so main can attach the
    // hand-init client after the hand SHM is opened (it is not available at initialize() time).
    std::shared_ptr<shm::ShmSettingsService> settingsService() const { return shm_settings_; }

    // Attach the settings service's CAN On/Off clients (bms/torque/initialize) to the services
    // embedded in the CAN segment (canShm), which the entrypoint opens after initialize(). Blocks,
    // retrying at 1 Hz, until the CAN node has published the services or shutdown is requested.
    bool attachCanServices(robros::shm::shm::canShm_st *can_shm);

    // Start accepting TCP client commands. Deferred from initialize() so the entrypoint can attach
    // all SHM service clients (attachCanServices/attachHandService) first; otherwise a command could be
    // dispatched while the CAN On/Off ServiceClients are still null.
    void startTcpServer();

  private:
    std::atomic<bool> running_{true};

    int argc_;
    char **argv_;
    igris_c::msg::dds::RobotEnvironment environment_;

    rclcpp::Node::SharedPtr node_;
    BridgeConfig bridge_cfg_;

    std::shared_ptr<DataStore> data_store_;
    std::shared_ptr<BridgeState> state_;
    std::shared_ptr<schema::CommandSchema> command_schema_;
    std::shared_ptr<shm::ShmCommandWriter> command_writer_;
    std::shared_ptr<tcp::TcpServer> tcp_server_;
    std::shared_ptr<dds::DdsManager> dds_manager_;
    std::shared_ptr<ros2::Ros2Manager> ros2_manager_;

    std::unique_ptr<robros::shm::shm::SharedMemoryManager<robros::shm::shm::controlShm_st>> spsc_manager_;
    robros::shm::shm::controlShm_st *spsc_ptr_ = nullptr;

    std::shared_ptr<ControlContext> control_ctx_;
    std::shared_ptr<MonitoringContext> monitor_ctx_;
    std::shared_ptr<SettingsContext> settings_ctx_;
    std::shared_ptr<shm::ShmSettingsService> shm_settings_;
};

// --- Template implementation ---

template <typename ReaderT> int BridgeMain::run(std::shared_ptr<ReaderT> sensor_reader) {
    sensor_reader->start();
    robros::shm::logger::get_logger("igris_c_bridge").info("SHM sensor reader started");

    // ROS2 spin in background
    auto ros_executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    ros_executor->add_node(node_);
    std::thread ros_spin_thr([&ros_executor]() { ros_executor->spin(); });

    // Wait for shutdown
    robros::shm::logger::get_logger("igris_c_bridge").info("Running. Press Ctrl+C to stop.");
    while (running_.load() && rclcpp::ok()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ros_executor->cancel();

    // Cleanup
    robros::shm::logger::get_logger("igris_c_bridge").info("Shutting down...");
    running_.store(false);

    sensor_reader->stop();
    if (ros2_manager_) {
        ros2_manager_->stop();
    }
    dds_manager_->stop();
    tcp_server_->stop();

    if (ros_spin_thr.joinable()) {
        ros_spin_thr.join();
    }

    // Clean up SPSC
    spsc_ptr_->magic.store(0, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    spsc_manager_.reset();

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }

    robros::shm::logger::get_logger("igris_c_bridge").info("Done.");
    return 0;
}

}  // namespace robros::bridge
