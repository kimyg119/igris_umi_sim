#ifndef __CONFIG_E27A7450_1DB7_41E0_8A62_65AEB1FCF2DA_H__
#define __CONFIG_E27A7450_1DB7_41E0_8A62_65AEB1FCF2DA_H__

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace robros::lib::util {

class Config {
  public:
    using ValueMap = std::unordered_map<std::string, std::string>;

    virtual ~Config() = default;

    virtual bool parse(const std::string &input) = 0;
    virtual std::string dump() const             = 0;

    bool has(const std::string &key) const { return data_.find(key) != data_.end(); }

    std::string get(const std::string &key, const std::string &default_value = "") const {
        auto it = data_.find(key);
        return (it == data_.end()) ? default_value : it->second;
    }

    void set(const std::string &key, const std::string &value) { data_[key] = value; }

  protected:
    ValueMap data_;
};

class YamlParser : public Config {
  public:
    bool parse(const std::string &input) override {
        data_.clear();
        try {
            YAML::Node root = YAML::Load(input);
            if (!root || !root.IsMap()) {
                return false;
            }

            for (const auto &item : root) {
                if (!item.first.IsScalar()) {
                    continue;
                }

                const std::string key = item.first.as<std::string>();
                if (item.second.IsScalar()) {
                    data_[key] = item.second.as<std::string>();
                } else {
                    YAML::Emitter emitter;
                    emitter << item.second;
                    data_[key] = emitter.c_str();
                }
            }
        } catch (...) {
            return false;
        }

        return !data_.empty();
    }

    std::string dump() const override {
        YAML::Node root(YAML::NodeType::Map);
        for (const auto &[key, value] : data_) {
            root[key] = value;
        }
        YAML::Emitter emitter;
        emitter << root;
        return emitter.c_str();
    }
};

class JsonParser : public Config {
  public:
    bool parse(const std::string &input) override {
        data_.clear();
        nlohmann::json root = nlohmann::json::parse(input, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            return false;
        }

        for (auto it = root.begin(); it != root.end(); ++it) {
            if (it.value().is_string()) {
                data_[it.key()] = it.value().get<std::string>();
            } else {
                data_[it.key()] = it.value().dump();
            }
        }

        return !data_.empty();
    }

    std::string dump() const override {
        nlohmann::json root = nlohmann::json::object();
        for (const auto &[key, value] : data_) {
            root[key] = value;
        }
        return root.dump();
    }
};

}  // namespace robros::lib::util

#endif /* __CONFIG_E27A7450_1DB7_41E0_8A62_65AEB1FCF2DA_H__ */
