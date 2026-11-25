// MuJoCo 시뮬레이션 제어 명령을 DDS로 보낸다.
// bridge_sim 이 rt/service/mujoco_sim/request 를 받아 mujoco SHM의 sim_service 채널에 쓴다.
//
//   mujoco_sim_ctl <pause|resume|reset|reload> [--domain-id N] [--namespace NS] [--cyclonedds-xml PATH]

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

igris_c::msg::dds::Header BuildHeader(uint32_t seq, const char *frame_id) {
  igris_c::msg::dds::Header header;
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

bool ParseCommand(const std::string &name,
                  igris_c::msg::dds::MujocoSimCommandType &out) {
  using T = igris_c::msg::dds::MujocoSimCommandType;
  if (name == "pause")  { out = T::MUJOCO_SIM_PAUSE;  return true; }
  if (name == "resume") { out = T::MUJOCO_SIM_RESUME; return true; }
  if (name == "reset")  { out = T::MUJOCO_SIM_RESET;  return true; }
  if (name == "reload") { out = T::MUJOCO_SIM_RELOAD; return true; }
  return false;
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "사용법: mujoco_sim_ctl <pause|resume|reset|reload>"
                 " [--domain-id N] [--namespace NS] [--cyclonedds-xml PATH]\n";
    return 1;
  }

  igris_c::msg::dds::MujocoSimCommandType command_type;
  if (!ParseCommand(argv[1], command_type)) {
    std::cerr << "알 수 없는 명령: " << argv[1] << "\n";
    return 1;
  }

  int domain_id = 0;
  std::string robot_namespace = "igris_c_001";
  std::string cyclonedds_xml_path;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--domain-id" && i + 1 < argc) { domain_id = std::stoi(argv[++i]); continue; }
    if (arg == "--namespace" && i + 1 < argc) { robot_namespace = argv[++i]; continue; }
    if (arg == "--cyclonedds-xml" && i + 1 < argc) { cyclonedds_xml_path = argv[++i]; continue; }
    std::cerr << "알 수 없는 인자: " << arg << "\n";
    return 1;
  }

  igris_c_sdk::ChannelFactory::Instance()->Init(domain_id, robot_namespace,
                                                ReadFile(cyclonedds_xml_path));
  if (!igris_c_sdk::ChannelFactory::Instance()->IsInitialized()) {
    std::cerr << "ChannelFactory 초기화 실패\n";
    return 1;
  }

  const std::string topic = "rt/service/mujoco_sim/request";
  auto publisher = std::make_unique<
      igris_c_sdk::Publisher<igris_c::msg::dds::MujocoSimCmd>>(
      topic, igris_c_sdk::QosProfile::Services());
  if (!publisher->init()) {
    std::cerr << "퍼블리셔 init 실패\n";
    return 1;
  }

  // 구독자 디스커버리를 기다린다. 서비스 QoS라도 매칭 전에 쓰면 유실된다.
  std::this_thread::sleep_for(std::chrono::milliseconds(800));

  igris_c::msg::dds::MujocoSimCmd request;
  request.header(BuildHeader(1, "mujoco_sim_ctl"));
  request.request_id(std::string("mujoco_sim_ctl_") + argv[1]);
  request.command_type(command_type);

  if (!publisher->write(request)) {
    std::cerr << "발행 실패\n";
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  std::cout << "발행 완료: " << argv[1] << "  topic="
            << igris_c_sdk::ChannelFactory::Instance()->resolve(topic) << "\n";
  return 0;
}
