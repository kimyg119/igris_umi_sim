#ifndef __SRC_IGRIS_C_CAN_INCLUDE_CAN_CAN_IMPL_H__
#define __SRC_IGRIS_C_CAN_INCLUDE_CAN_CAN_IMPL_H__

#include "can_handler.h"
#include "dummycan_controller.h"
#include "pcan_controller.h"

namespace robros {
namespace can {

typedef CANHandler<PCANController> PCANMotorHandler;
typedef CANHandler<DummyCANController> DummyCANMotorHandler;

}  // namespace can
}  // namespace robros

#endif /* __SRC_IGRIS_C_CAN_INCLUDE_CAN_CAN_IMPL_H__ */
