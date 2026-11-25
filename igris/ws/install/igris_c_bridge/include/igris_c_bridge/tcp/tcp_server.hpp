#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "igris_c_bridge/data_store.hpp"
#include "igris_c_bridge/schema/command_schema.hpp"
#include "igris_c_bridge/types.hpp"

namespace robros::bridge {

class IApiHandler;  // fwd

namespace tcp {

class TcpServer {
  public:
    TcpServer(int port, std::string bind_ip, std::shared_ptr<DataStore> data_store, std::shared_ptr<schema::CommandSchema> schema = nullptr,
              int timeout_ms = 10000);
    ~TcpServer();

    bool start();
    void stop();

    // Handler registration
    void registerHandler(std::shared_ptr<IApiHandler> handler);

    // Response utilities
    void sendResponse(int fd, const json &body);
    void sendSuccess(int fd, const std::string &requestId, const json &data = json::object(), const std::string &message = "ok");
    void sendError(int fd, const std::string &requestId, const std::string &code, const std::string &message);

    // Async completion (called from executor callback, routes response to correct fd)
    void notifyComplete(const std::string &requestId, bool success, const std::string &error_code = "", const std::string &msg = "ok",
                        const json &data = json::object());

    // Request parsing
    bool parseRequest(const json &j, RequestCommon &out, std::string &why);
    void handleLine(const std::string &line, int client_fd);

    // Data store access
    DataStore &dataStore() { return *data_store_; }
    const DataStore &dataStore() const { return *data_store_; }

  private:
    void serverThread();

    int port_;
    std::string bind_ip_;

    int server_fd_{-1};
    std::thread server_thr_;
    std::atomic<bool> running_{false};

    // functionType -> handler
    std::mutex handlers_mtx_;
    std::map<std::string, std::shared_ptr<IApiHandler>> handlers_;

    // Send synchronization
    std::mutex send_mtx_;

    // requestId -> socket FD mapping (for async response routing)
    std::mutex reqmap_mtx_;
    std::unordered_map<std::string, int> req_to_fd_;

    // socket FD -> HTTP protocol detection
    std::mutex fd_protocol_mtx_;
    std::unordered_map<int, bool> fd_is_http_;

    // Shared data store
    std::shared_ptr<DataStore> data_store_;

    // Command schema for validation
    std::shared_ptr<schema::CommandSchema> schema_;
};

}  // namespace tcp
}  // namespace robros::bridge
