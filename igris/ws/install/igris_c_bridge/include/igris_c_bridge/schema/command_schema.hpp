#pragma once

#include <map>
#include <nlohmann/json-schema.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "igris_c_lib/util/schema.h"

namespace robros::bridge::schema {

using json = nlohmann::json;

struct CommandMeta : robros::lib::util::SchemaStruct<CommandMeta> {
    std::string functionType;
    std::optional<int> timeout_sec;

    static constexpr auto schemaFields() {
        return std::make_tuple(robros::lib::util::makeField("functionType", &CommandMeta::functionType, true),
                               robros::lib::util::makeField("timeout_sec", &CommandMeta::timeout_sec, false));
    }
};

struct ValidationResult {
    bool valid = true;
    std::string error_code;
    std::string error_message;
};

struct ImuErrorMapping {
    int code;
    std::map<std::string, std::string> flags;  // flag_name -> error_message
};

// Loads JSON schema files at startup and provides:
// - Command registry (valid commands, metadata)
// - commandBody validation via JSON Schema draft-07
// - Error code/message mappings
class CommandSchema {
  public:
    explicit CommandSchema(const std::string &schemas_dir);

    // Command registry
    bool isValidCommand(const std::string &commandType) const;
    CommandMeta getCommandMeta(const std::string &commandType) const;
    std::string getFunctionType(const std::string &commandType) const;
    std::vector<std::string> getCommandsForFunction(const std::string &functionType) const;

    // commandBody validation using JSON Schema
    ValidationResult validateCommandBody(const std::string &commandType, const json &body) const;

    // Error code mappings
    const std::map<uint32_t, std::string> &getMotorErrorMap() const;
    int getEstopCode() const;
    const std::string &getEstopMessage() const;
    const ImuErrorMapping &getImuErrorMapping() const;
    int getMotorErrorCode() const;

  private:
    void loadCommands(const std::string &path);
    void loadControlSchemas(const std::string &path);
    void loadErrorCodes(const std::string &path);

    json loadJsonFile(const std::string &path) const;

    // Command registry
    std::unordered_map<std::string, CommandMeta> commands_;

    // JSON Schema validators for commandBody (per commandType)
    std::unordered_map<std::string, nlohmann::json_schema::json_validator> body_validators_;

    // Error mappings
    std::map<uint32_t, std::string> motor_error_map_;
    int estop_code_ = 3000;
    std::string estop_message_;
    ImuErrorMapping imu_error_mapping_;
    int motor_error_code_ = 4000;
};

}  // namespace robros::bridge::schema
