#pragma once

#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "igris_c_bridge/bridge_state.hpp"
#include "igris_c_bridge/schema/command_schema.hpp"
#include "igris_c_bridge/types.hpp"
#include "igris_c_shm/service/service.h"
#include "igris_c_shm/shm/types.h"

namespace robros::bridge::shm {

// Writes control commands to controller via service (request/response) or SPSC (fire-and-forget).
// Service commands get async completion callbacks; SPSC commands are fire-and-forget.
class ShmCommandWriter : public IControlExecutor {
  public:
    ShmCommandWriter(std::shared_ptr<BridgeState> state, robros::shm::shm::controlShm_st *spsc_shm,
                     std::shared_ptr<schema::CommandSchema> schema);
    ~ShmCommandWriter() override;

    // Attach the ServiceClient to the bridge -> controller command service embedded in the
    // controlShm segment this writer holds (spsc_shm_). Returns false until the controller has
    // published the service magic (caller retries).
    bool attachServiceClient();

    void submitAsync(const std::string &requestId, const std::string &commandType, const json &commandBody,
                     std::function<void(bool ok, const std::string &error_code, const std::string &msg, const json &data)> cb) override;

    json statusJson() override;

  private:
    static std::string to_uppercase(const std::string &str);
    bool is_valid_preset(const std::string &preset) const;

    std::shared_ptr<BridgeState> state_;
    robros::shm::shm::controlShm_st *spsc_shm_;
    std::shared_ptr<schema::CommandSchema> schema_;

    std::shared_ptr<
        robros::shm::service::ServiceClient<robros::shm::service::BridgeServiceRequest_st, robros::shm::service::BridgeServiceResponse_st>>
        service_client_;

    std::mutex futures_mtx_;
    std::vector<std::future<void>> pending_futures_;
    void cleanupFinishedFutures();
};

}  // namespace robros::bridge::shm
