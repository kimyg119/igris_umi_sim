#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "igris_c_bridge/types.hpp"

namespace robros::bridge {
namespace tcp {
class TcpServer;
}

namespace schema {

// Generic commandType -> handler function dispatcher.
// Replaces if-else chains in API handlers.
class CommandRouter {
  public:
    using HandlerFn = std::function<void(const RequestCommon &, tcp::TcpServer &, int)>;

    void registerCommand(const std::string &commandType, HandlerFn fn);
    bool dispatch(const RequestCommon &rc, tcp::TcpServer &server, int fd) const;

  private:
    std::unordered_map<std::string, HandlerFn> handlers_;
};

}  // namespace schema
}  // namespace robros::bridge
