// IGRIS-C 로컬 시뮬 제어 도구. igris_c_sdk 로 DDS 요청을 직접 발행한다.
//
//   igris_ctl sim    <pause|resume|reset|reload>
//   igris_ctl torque <on|off>
//   igris_ctl mode   <low|lowwalk|hold|stop|walk|home|preset:<NAME>>
//
// 공통 옵션: --domain-id N  --namespace NS  --cyclonedds-xml PATH

#include "igris_c_sdk/channel_factory.hpp"
#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/publisher.hpp"
#include "igris_c_sdk/qos.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {

namespace msgs = igris_c::msg::dds;

msgs::Header BuildHeader(uint32_t seq, const char *frame_id) {
  msgs::Header header;
  header.seq(seq);
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  header.sec(static_cast<uint32_t>(ns / 1000000000ULL));
  header.nanosec(static_cast<uint32_t>(ns % 1000000000ULL));
  std::array<char, 256> frame_id_array{};
  for (std::size_t i = 0; i < frame_id_array.size() - 1 && frame_id[i] != '\0'; ++i) {
    frame_id_array[i] = frame_id[i];
  }
  header.frame_id(frame_id_array);
  return header;
}

std::string ReadFile(const std::string &path) {
  if (path.empty()) return "";
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "CycloneDDS XML을 열 수 없음: " << path << "\n";
    return "";
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

// 구독자 디스커버리 전에 쓰면 유실되므로 잠시 기다렸다 발행한다.
template <typename MsgT>
bool PublishOnce(const std::string &topic, const MsgT &message) {
  auto publisher = std::make_unique<igris_c_sdk::Publisher<MsgT>>(
      topic, igris_c_sdk::QosProfile::Services());
  if (!publisher->init()) {
    std::cerr << "퍼블리셔 init 실패: " << topic << "\n";
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  if (!publisher->write(message)) {
    std::cerr << "발행 실패: " << topic << "\n";
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  return true;
}

void Usage() {
  std::cout <<
    "사용법:\n"
    "  igris_ctl sim    <pause|resume|reset|reload>\n"
    "  igris_ctl torque <on|off>\n"
    "  igris_ctl mode   <low|lowwalk|hold|stop|walk|home|preset:NAME>\n"
    "\n옵션: --domain-id N  --namespace NS  --cyclonedds-xml PATH\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) { Usage(); return 1; }

  const std::string group = argv[1];
  const std::string action = argv[2];

  int domain_id = 0;
  std::string robot_namespace = "igris_c_001";
  std::string cyclonedds_xml_path;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--domain-id" && i + 1 < argc) { domain_id = std::stoi(argv[++i]); continue; }
    if (arg == "--namespace" && i + 1 < argc) { robot_namespace = argv[++i]; continue; }
    if (arg == "--cyclonedds-xml" && i + 1 < argc) { cyclonedds_xml_path = argv[++i]; continue; }
    std::cerr << "알 수 없는 인자: " << arg << "\n"; return 1;
  }

  igris_c_sdk::ChannelFactory::Instance()->Init(domain_id, robot_namespace,
                                                ReadFile(cyclonedds_xml_path));
  if (!igris_c_sdk::ChannelFactory::Instance()->IsInitialized()) {
    std::cerr << "ChannelFactory 초기화 실패\n";
    return 1;
  }

  if (group == "sim") {
    msgs::MujocoSimCommandType type;
    if      (action == "pause")  type = msgs::MujocoSimCommandType::MUJOCO_SIM_PAUSE;
    else if (action == "resume") type = msgs::MujocoSimCommandType::MUJOCO_SIM_RESUME;
    else if (action == "reset")  type = msgs::MujocoSimCommandType::MUJOCO_SIM_RESET;
    else if (action == "reload") type = msgs::MujocoSimCommandType::MUJOCO_SIM_RELOAD;
    else { std::cerr << "알 수 없는 sim 명령: " << action << "\n"; return 1; }

    msgs::MujocoSimCmd request;
    request.header(BuildHeader(1, "igris_ctl"));
    request.request_id("igris_ctl_sim_" + action);
    request.command_type(type);
    if (!PublishOnce("rt/service/mujoco_sim/request", request)) return 1;

  } else if (group == "torque") {
    msgs::TorqueType type;
    if      (action == "on")  type = msgs::TorqueType::TORQUE_ON;
    else if (action == "off") type = msgs::TorqueType::TORQUE_OFF;
    else { std::cerr << "알 수 없는 torque 명령: " << action << "\n"; return 1; }

    msgs::TorqueCmd request;
    request.header(BuildHeader(1, "igris_ctl"));
    request.request_id("igris_ctl_torque_" + action);
    request.torque(type);
    if (!PublishOnce("rt/service/torque/request", request)) return 1;

  } else if (group == "mode") {
    msgs::ControlModeCommandType type;
    std::string preset_id;
    if      (action == "low")  type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL;
    else if (action == "hold") type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_JOINT_POSITION_HOLD;
    else if (action == "stop") type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_MOTION_STOP;
    else if (action == "walk") type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_WALKMODE_ON;
    else if (action == "lowwalk") type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_LOW_LEVEL_WALKMODE_ON;
    else if (action == "home") {
      type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_MOTION_PRESET;
      preset_id = "HOME";
    } else if (action.rfind("preset:", 0) == 0) {
      type = msgs::ControlModeCommandType::CONTROL_MODE_CMD_MOTION_PRESET;
      preset_id = action.substr(7);
    } else { std::cerr << "알 수 없는 mode 명령: " << action << "\n"; return 1; }

    msgs::ControlModeCommandRequest request;
    request.header(BuildHeader(1, "igris_ctl"));
    request.request_id("igris_ctl_mode_" + action);
    request.command_type(type);
    request.preset_id(preset_id);
    request.is_cyclic(false);
    if (!PublishOnce("rt/service/control_mode/request", request)) return 1;

  } else {
    Usage(); return 1;
  }

  std::cout << "발행 완료: " << group << " " << action << "\n";
  return 0;
}
