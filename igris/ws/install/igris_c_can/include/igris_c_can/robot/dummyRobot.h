#ifndef __DUMMY_BC897FA1_7B23_4776_89C5_034AF1997528_H__
#define __DUMMY_BC897FA1_7B23_4776_89C5_034AF1997528_H__

#include "igris_c_lib/util/task.h"
#include "igris_c_shm/logger/logger.h"

#include <string>
#include <vector>

// Forward declaration: canShm() returns a pointer, so the full type is only needed in the .cpp
// (which includes igris_c_shm/shm/types.h).
namespace robros::shm::shm {
struct canShm_st;
}

namespace robros::can {
template <class CanHandler> class DummyRobot {
  public:
    DummyRobot();
    ~DummyRobot();

    int initializeHardware();
    int powerOn();
    int powerOff();
    int torqueOn();
    int torqueOff();
    std::vector<std::string> run();

  private:
    void fetchCommand();
    void pushState();
    void control(int channel);
    void workingChannel(int channel);

  public:
    int execute();

    // The dummy CAN segment (created in init). Exposed so the dummy CAN node can attach the
    // embedded On/Off service servers (bms/torque/initialize) after construction.
    robros::shm::shm::canShm_st *canShm() const;

  private:
    robros::lib::util::RtSingleTask *commandFetchTask_;
    robros::lib::util::RtSingleTask *pushStateTask_;
    robros::lib::util::RtSingleTask *channelControlTasks_[4];
};
}  // namespace robros::can

#endif /* __DUMMY_BC897FA1_7B23_4776_89C5_034AF1997528_H__ */
