#pragma once
#include <memory>
#include <nlohmann/json.hpp>

#include "igris_c_bridge/schema/command_router.hpp"
#include "igris_c_bridge/schema/command_schema.hpp"
#include "igris_c_bridge/types.hpp"

namespace robros::bridge {
namespace tcp {
class TcpServer;  // fwd
}

struct IApiHandler {
    virtual ~IApiHandler()                                                      = default;
    virtual const char *function_key() const                                    = 0;
    virtual void handle(const RequestCommon &, tcp::TcpServer &, int client_fd) = 0;
};

// functionType="control"
class ControlApiHandler : public IApiHandler {
  public:
    ControlApiHandler(std::shared_ptr<ControlContext> ctx, std::shared_ptr<schema::CommandSchema> schema)
        : ctx_(std::move(ctx)), schema_(std::move(schema)) {}
    const char *function_key() const override { return "control"; }
    void handle(const RequestCommon &rc, tcp::TcpServer &server, int client_fd) override;

  private:
    std::shared_ptr<ControlContext> ctx_;
    std::shared_ptr<schema::CommandSchema> schema_;
};

// functionType="monitoring"
class MonitoringApiHandler : public IApiHandler {
  public:
    MonitoringApiHandler(std::shared_ptr<MonitoringContext> ctx, std::shared_ptr<schema::CommandSchema> schema);
    const char *function_key() const override { return "monitoring"; }
    void handle(const RequestCommon &rc, tcp::TcpServer &server, int fd) override;

  private:
    void registerCommands();
    nlohmann::json getMotionList();

    std::shared_ptr<MonitoringContext> ctx_;
    std::shared_ptr<schema::CommandSchema> schema_;
    schema::CommandRouter router_;
};

// functionType="settings"
class SettingsApiHandler : public IApiHandler {
  public:
    SettingsApiHandler(std::shared_ptr<SettingsContext> ctx, std::shared_ptr<schema::CommandSchema> schema);
    const char *function_key() const override { return "setting"; }
    void handle(const RequestCommon &rc, tcp::TcpServer &server, int fd) override;

  private:
    void registerCommands();
    void handleTorque(bool enable, const RequestCommon &rc, tcp::TcpServer &server, int fd);
    void handleMotor(bool enable, const RequestCommon &rc, tcp::TcpServer &server, int fd);
    void handleHandInit(const RequestCommon &rc, tcp::TcpServer &server, int fd);

    std::shared_ptr<SettingsContext> ctx_;
    std::shared_ptr<schema::CommandSchema> schema_;
    schema::CommandRouter router_;
};

}  // namespace robros::bridge
