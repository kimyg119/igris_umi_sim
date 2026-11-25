/**
 * @file schema.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2026-03-14
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __SCHEMA_AFD083EF_D54C_4AFD_82AF_898110A1B452_H__
#define __SCHEMA_AFD083EF_D54C_4AFD_82AF_898110A1B452_H__

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <yaml-cpp/yaml.h>

namespace robros::lib::util {

class JsonSchema;
class YamlSchema;

template <typename T> struct is_optional : std::false_type {};
template <typename U> struct is_optional<std::optional<U>> : std::true_type {};

template <typename Owner, typename Member> struct FieldSpec {
    const char *name;
    Member Owner::*member;
    bool required;
};

template <typename Owner, typename Member>
constexpr FieldSpec<Owner, Member> makeField(const char *name, Member Owner::*member, bool required = true) {
    return FieldSpec<Owner, Member>{name, member, required};
}

namespace detail {
template <typename T> bool parseBoolString(const std::string &raw, T *out) {
    if (raw == "true" || raw == "1") {
        *out = true;
        return true;
    }
    if (raw == "false" || raw == "0") {
        *out = false;
        return true;
    }
    return false;
}

template <typename T> bool jsonToValue(const nlohmann::json &value, T *out) {
    if constexpr (is_optional<T>::value) {
        using U = typename T::value_type;
        if (value.is_null()) {
            *out = std::nullopt;
            return true;
        }
        U tmp{};
        if (!jsonToValue(value, &tmp)) {
            return false;
        }
        *out = tmp;
        return true;
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!value.is_string()) {
            return false;
        }
        *out = value.get<std::string>();
        return true;
    } else if constexpr (std::is_same_v<T, bool>) {
        if (value.is_boolean()) {
            *out = value.get<bool>();
            return true;
        }
        if (value.is_string()) {
            return parseBoolString(value.get<std::string>(), out);
        }
        return false;
    } else if constexpr (std::is_integral_v<T>) {
        if (value.is_number_integer()) {
            *out = static_cast<T>(value.get<long long>());
            return true;
        }
        if (value.is_string()) {
            try {
                *out = static_cast<T>(std::stoll(value.get<std::string>()));
                return true;
            } catch (...) {
                return false;
            }
        }
        return false;
    } else if constexpr (std::is_floating_point_v<T>) {
        if (value.is_number()) {
            *out = static_cast<T>(value.get<double>());
            return true;
        }
        if (value.is_string()) {
            try {
                *out = static_cast<T>(std::stod(value.get<std::string>()));
                return true;
            } catch (...) {
                return false;
            }
        }
        return false;
    } else if constexpr (std::is_same_v<T, nlohmann::json>) {
        *out = value;
        return true;
    } else {
        return false;
    }
}

template <typename T> nlohmann::json valueToJson(const T &value) {
    if constexpr (is_optional<T>::value) {
        if (!value.has_value()) {
            return nullptr;
        }
        return valueToJson(*value);
    } else if constexpr (std::is_same_v<T, nlohmann::json>) {
        return value;
    } else {
        return nlohmann::json(value);
    }
}

template <typename T> bool yamlToValue(const YAML::Node &node, T *out) {
    if constexpr (is_optional<T>::value) {
        using U = typename T::value_type;
        if (!node || node.IsNull()) {
            *out = std::nullopt;
            return true;
        }
        U tmp{};
        if (!yamlToValue(node, &tmp)) {
            return false;
        }
        *out = tmp;
        return true;
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!node.IsScalar()) {
            return false;
        }
        *out = node.as<std::string>();
        return true;
    } else if constexpr (std::is_same_v<T, bool>) {
        if (!node.IsScalar()) {
            return false;
        }
        const std::string raw = node.as<std::string>();
        return parseBoolString(raw, out);
    } else if constexpr (std::is_integral_v<T>) {
        if (!node.IsScalar()) {
            return false;
        }
        try {
            *out = static_cast<T>(std::stoll(node.as<std::string>()));
            return true;
        } catch (...) {
            return false;
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!node.IsScalar()) {
            return false;
        }
        try {
            *out = static_cast<T>(std::stod(node.as<std::string>()));
            return true;
        } catch (...) {
            return false;
        }
    } else {
        return false;
    }
}

template <typename T> YAML::Node valueToYaml(const T &value) {
    YAML::Node node;
    if constexpr (is_optional<T>::value) {
        if (value.has_value()) {
            node = valueToYaml(*value);
        }
    } else {
        node = value;
    }
    return node;
}
}  // namespace detail

template <typename Derived> class SchemaStruct {
  public:
    bool toJsonString(std::string *output, std::string *error = nullptr) const;
    bool toYamlString(std::string *output, std::string *error = nullptr) const;
};

class Schema {
  public:
    struct Rule {
        std::string type;
        bool required;
    };

    virtual ~Schema() = default;

    void addRule(const std::string &key, const std::string &type, bool required = true) {
        rules_[key] = Rule{normalizeType(type), required};
    }

    bool hasRule(const std::string &key) const { return rules_.find(key) != rules_.end(); }

    virtual bool apply(const std::string &input, std::string *error = nullptr) const = 0;

  protected:
    static std::string normalizeType(const std::string &type) {
        std::string out = type;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }

    static void setError(std::string *error, const std::string &message) {
        if (error != nullptr) {
            *error = message;
        }
    }

    std::unordered_map<std::string, Rule> rules_;
};

class JsonSchema : public Schema {
  public:
    bool apply(const std::string &input, std::string *error = nullptr) const override {
        nlohmann::json root = nlohmann::json::parse(input, nullptr, false);
        if (root.is_discarded()) {
            setError(error, "invalid json");
            return false;
        }
        if (!root.is_object()) {
            setError(error, "json root is not object");
            return false;
        }

        for (const auto &[key, rule] : rules_) {
            if (!root.contains(key)) {
                if (rule.required) {
                    setError(error, "missing required key: " + key);
                    return false;
                }
                continue;
            }

            const auto &value = root.at(key);
            if (!isValidType(value, rule.type)) {
                setError(error, "type mismatch for key: " + key + " (expected " + rule.type + ")");
                return false;
            }
        }

        return true;
    }

    template <typename T> bool parse(const std::string &input, T *out, std::string *error = nullptr) const {
        nlohmann::json root = nlohmann::json::parse(input, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            setError(error, "invalid json");
            return false;
        }

        T parsed{};
        bool ok = true;
        std::apply(
            [&](const auto &...field) {
                auto parse_field = [&](const auto &f) {
                    using FieldType = std::decay_t<decltype(parsed.*(f.member))>;
                    if (!root.contains(f.name)) {
                        if (f.required && !is_optional<FieldType>::value) {
                            setError(error, std::string("missing required key: ") + f.name);
                            return false;
                        }
                        if constexpr (is_optional<FieldType>::value) {
                            parsed.*(f.member) = std::nullopt;
                        }
                        return true;
                    }
                    if (!detail::jsonToValue(root.at(f.name), &(parsed.*(f.member)))) {
                        setError(error, std::string("type conversion failed for key: ") + f.name);
                        return false;
                    }
                    return true;
                };
                ((ok = ok && parse_field(field)), ...);
            },
            T::schemaFields());

        if (!ok) {
            return false;
        }
        if (out != nullptr) {
            *out = std::move(parsed);
        }
        return true;
    }

    template <typename T> bool dumpAuto(const T &data, std::string *output, std::string *error = nullptr) const {
        try {
            nlohmann::json root = nlohmann::json::object();
            std::apply(
                [&](const auto &...field) {
                    auto emit_field = [&](const auto &f) {
                        const auto &v   = data.*(f.member);
                        using FieldType = std::decay_t<decltype(v)>;
                        if constexpr (is_optional<FieldType>::value) {
                            if (!v.has_value()) {
                                return;
                            }
                        }
                        root[f.name] = detail::valueToJson(v);
                    };
                    (emit_field(field), ...);
                },
                T::schemaFields());

            if (output != nullptr) {
                *output = root.dump();
            }
        } catch (const std::exception &e) {
            setError(error, std::string("json auto dump failed: ") + e.what());
            return false;
        }
        return true;
    }

  private:
    static bool isValidType(const nlohmann::json &value, const std::string &type) {
        if (type == "any") {
            return true;
        }
        if (type == "string") {
            return value.is_string();
        }
        if (type == "number") {
            return value.is_number();
        }
        if (type == "integer") {
            return value.is_number_integer();
        }
        if (type == "boolean" || type == "bool") {
            return value.is_boolean();
        }
        if (type == "object") {
            return value.is_object();
        }
        if (type == "array") {
            return value.is_array();
        }
        if (type == "null") {
            return value.is_null();
        }
        return false;
    }
};

class YamlSchema : public Schema {
  public:
    bool apply(const std::string &input, std::string *error = nullptr) const override {
        YAML::Node root;
        try {
            root = YAML::Load(input);
        } catch (...) {
            setError(error, "invalid yaml");
            return false;
        }

        if (!root || !root.IsMap()) {
            setError(error, "yaml root is not map");
            return false;
        }

        for (const auto &[key, rule] : rules_) {
            const YAML::Node value = root[key];
            if (!value) {
                if (rule.required) {
                    setError(error, "missing required key: " + key);
                    return false;
                }
                continue;
            }

            if (!isValidType(value, rule.type)) {
                setError(error, "type mismatch for key: " + key + " (expected " + rule.type + ")");
                return false;
            }
        }

        return true;
    }

    template <typename T> bool parse(const std::string &input, T *out, std::string *error = nullptr) const {
        YAML::Node root;
        try {
            root = YAML::Load(input);
        } catch (...) {
            setError(error, "invalid yaml");
            return false;
        }
        if (!root || !root.IsMap()) {
            setError(error, "yaml root is not map");
            return false;
        }

        T parsed{};
        bool ok = true;
        std::apply(
            [&](const auto &...field) {
                auto parse_field = [&](const auto &f) {
                    using FieldType       = std::decay_t<decltype(parsed.*(f.member))>;
                    const YAML::Node node = root[f.name];
                    if (!node) {
                        if (f.required && !is_optional<FieldType>::value) {
                            setError(error, std::string("missing required key: ") + f.name);
                            return false;
                        }
                        if constexpr (is_optional<FieldType>::value) {
                            parsed.*(f.member) = std::nullopt;
                        }
                        return true;
                    }
                    if (!detail::yamlToValue(node, &(parsed.*(f.member)))) {
                        setError(error, std::string("type conversion failed for key: ") + f.name);
                        return false;
                    }
                    return true;
                };
                ((ok = ok && parse_field(field)), ...);
            },
            T::schemaFields());

        if (!ok) {
            return false;
        }
        if (out != nullptr) {
            *out = std::move(parsed);
        }
        return true;
    }

    template <typename T> bool dumpAuto(const T &data, std::string *output, std::string *error = nullptr) const {
        try {
            YAML::Node root(YAML::NodeType::Map);
            std::apply(
                [&](const auto &...field) {
                    auto emit_field = [&](const auto &f) {
                        const auto &v   = data.*(f.member);
                        using FieldType = std::decay_t<decltype(v)>;
                        if constexpr (is_optional<FieldType>::value) {
                            if (!v.has_value()) {
                                return;
                            }
                        }
                        root[f.name] = detail::valueToYaml(v);
                    };
                    (emit_field(field), ...);
                },
                T::schemaFields());

            YAML::Emitter emitter;
            emitter << root;
            if (output != nullptr) {
                *output = emitter.c_str();
            }
        } catch (const std::exception &e) {
            setError(error, std::string("yaml auto dump failed: ") + e.what());
            return false;
        }
        return true;
    }

  private:
    static bool isValidType(const YAML::Node &value, const std::string &type) {
        if (type == "any") {
            return true;
        }
        if (type == "string") {
            return value.IsScalar();
        }
        if (type == "number") {
            if (!value.IsScalar()) {
                return false;
            }
            try {
                (void)value.as<double>();
                return true;
            } catch (...) {
                return false;
            }
        }
        if (type == "integer") {
            if (!value.IsScalar()) {
                return false;
            }
            try {
                (void)value.as<long long>();
                return true;
            } catch (...) {
                return false;
            }
        }
        if (type == "boolean" || type == "bool") {
            if (!value.IsScalar()) {
                return false;
            }
            try {
                (void)value.as<bool>();
                return true;
            } catch (...) {
                return false;
            }
        }
        if (type == "object") {
            return value.IsMap();
        }
        if (type == "array") {
            return value.IsSequence();
        }
        if (type == "null") {
            return value.IsNull();
        }
        return false;
    }
};

template <typename Derived> bool SchemaStruct<Derived>::toJsonString(std::string *output, std::string *error) const {
    JsonSchema schema;
    return schema.dumpAuto<Derived>(static_cast<const Derived &>(*this), output, error);
}

template <typename Derived> bool SchemaStruct<Derived>::toYamlString(std::string *output, std::string *error) const {
    YamlSchema schema;
    return schema.dumpAuto<Derived>(static_cast<const Derived &>(*this), output, error);
}

}  // namespace robros::lib::util

#endif /* __SCHEMA_AFD083EF_D54C_4AFD_82AF_898110A1B452_H__ */
