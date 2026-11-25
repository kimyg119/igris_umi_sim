#ifndef __COMMON_AD83175D_483E_4082_ACD1_C3512267A621_H__
#define __COMMON_AD83175D_483E_4082_ACD1_C3512267A621_H__

#include <chrono>
#include <cstdint>
#include <cstring>

namespace robros::shm::common {
struct Header {
    uint32_t seq;
    uint32_t sec;
    uint32_t nanosec;
    char frame_id[256];

    Header() : seq(0), sec(0), nanosec(0) { std::memset(frame_id, 0, sizeof(frame_id)); }

    void update() {
        seq++;
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto ns  = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        sec            = static_cast<uint32_t>(ns / 1000000000ULL);
        nanosec        = static_cast<uint32_t>(ns % 1000000000ULL);
    }
};

constexpr std::size_t MUJOCO_MAX_JOINT   = 50;
constexpr std::size_t HAND_V1_JOINT_NUM  = 12;
constexpr std::size_t PARALLEL_JOINT_NUM = 5;

}  // namespace robros::shm::common

#endif /* __COMMON_AD83175D_483E_4082_ACD1_C3512267A621_H__ */
