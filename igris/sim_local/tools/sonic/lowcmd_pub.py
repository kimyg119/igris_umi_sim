"""cyclonedds LowCmd publisher for the IGRIS-C SDK native DDS command path.

Publishes ``igris_c::msg::dds::LowCmd`` to the native topic ``igris_c_001/rt/lowcmd``
(best_effort / volatile / keep_last-5) — the exact topic + type + QoS the SDK
controller reads. Verified against:
  - igris_c_gmt_public/src/io/robot_io.cpp        (Publisher<LowCmd>("rt/lowcmd", SensorData))
  - igris_c_sdk_public/cyclonedds/cyclonedds_low_level_control.cpp  (motor.id(i), PJS fill)
  - live DDS discovery of the controller's reader (type igris_c::msg::dds::LowCmd)

VERSION (critical): the SDK / ROS 2 stack runs CycloneDDS 0.10.5. Use a
version-matched cyclonedds-python (0.10.5). A newer 11.x wheel advertises an
XTypes type object the 0.10.5 rmw_cyclonedds cannot parse and SEGFAULTS the ROS
nodes (igris_c_gmt_public) on discovery — observed. See sim_local/Dockerfile.

The type MUST match the NATIVE C++ type in igris_c_msgs.hpp exactly (the SDK
reader is DisallowTypeCoercion). Non-obvious fields — confirmed by watching the
controller report "LowCmd: 50 Hz" only once these were right:
  - Header.frame_id is a fixed char[256] array, NOT an unbounded string.
  - kinematic_modes is an array of the 32-bit enum KinematicMode {MS,PJS},
    NOT uint8[5]  (the rosidl-adapter .idl is misleading; the native type wins).
Types are @final + @autoid("sequential") with dotted typenames so the XTypes
typeid matches the C++ idlc output.
"""
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic

NUM_MOTORS = 31
TOPIC = "igris_c_001/rt/lowcmd"
_FRAME_ID = ["\x00"] * 256  # native Header.frame_id is char[256]

# 31 system-joint names in JointIndex order (types.hpp) — this is also exactly
# the order /mujoco/joint_states publishes, so a name->position gather in this
# order yields the SDK 31-motor vector directly.
SYS_JOINT_NAMES = [
    "waist_yaw", "l_waist_flange", "r_waist_flange",                       # 0 WY, 1 WR, 2 WP
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw",                              # 3-5  LHP LHR LHY
    "l_knee_pitch", "l_ankle_flange_out", "l_ankle_flange_in",            # 6-8  LKP LAP LAR
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw",                              # 9-11 RHP RHR RHY
    "r_knee_pitch", "r_ankle_flange_out", "r_ankle_flange_in",            # 12-14 RKP RAP RAR
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow_pitch",  # 15-18 LSP LSR LSY LEP
    "l_wrist_yaw", "l_wrist_motor_output_front", "l_wrist_motor_output_back",  # 19-21 L-wrist
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch",  # 22-25 RSP RSR RSY REP
    "r_wrist_yaw", "r_wrist_motor_output_front", "r_wrist_motor_output_back",  # 26-28 R-wrist
    "neck_yaw", "neck_pitch",                                              # 29-30 neck yaw/pitch
]


class KinematicMode(idl.IdlEnum, typename="igris_c.msg.dds.KinematicMode"):
    MS = 0
    PJS = 1


PJS = KinematicMode.PJS  # igris_c::msg::dds::KinematicMode::PJS (parallel joint space)


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Header(idl.IdlStruct, typename="igris_c.msg.dds.Header"):
    seq: types.uint32
    sec: types.uint32
    nanosec: types.uint32
    frame_id: types.array[types.char, 256]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd(idl.IdlStruct, typename="igris_c.msg.dds.MotorCmd"):
    id: types.uint16
    q: types.float32
    dq: types.float32
    tau: types.float32
    kp: types.float32
    kd: types.float32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd(idl.IdlStruct, typename="igris_c.msg.dds.LowCmd"):
    header: Header
    kinematic_modes: types.array[KinematicMode, 5]
    motors: types.array[MotorCmd, 31]


class LowCmdPublisher:
    """Best-effort cyclonedds writer for igris_c_001/rt/lowcmd."""

    def __init__(self, domain: int = 0):
        self.dp = DomainParticipant(domain)
        qos = Qos(Policy.Reliability.BestEffort,
                  Policy.Durability.Volatile,
                  Policy.History.KeepLast(5))
        self.tp = Topic(self.dp, TOPIC, LowCmd, qos=qos)
        self.dw = DataWriter(Publisher(self.dp), self.tp, qos=qos)
        self._seq = 0

    def send(self, q31, dq31, kp31, kd31, kinematic_modes: KinematicMode = PJS):
        """Write one LowCmd. All args are length-31 sequences (SDK motor order)."""
        assert len(q31) == len(dq31) == len(kp31) == len(kd31) == NUM_MOTORS, \
            "q/dq/kp/kd must all be length 31"
        motors = [MotorCmd(id=i, q=float(q31[i]), dq=float(dq31[i]), tau=0.0,
                           kp=float(kp31[i]), kd=float(kd31[i]))
                  for i in range(NUM_MOTORS)]
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self.dw.write(LowCmd(header=Header(self._seq, 0, 0, _FRAME_ID),
                             kinematic_modes=[kinematic_modes] * 5,
                             motors=motors))


class LowCmdPublisherROS:
    """SDK-ROS-path lowcmd publisher: igris_c_sdk/msg/LowCmd on ROS topic igris_c_001/rt/lowcmd.

    Why this exists (2026-08-31): the sim's NATIVE LowCmd reader declares
    TypeConsistency.DisallowTypeCoercion — with XTypes-era cyclonedds-python (>=0.10.5/11.x) our
    self-defined IdlStruct ships a type object that fails that exact-match check, so the raw
    LowCmdPublisher above publishes into the void (S2-era bindings predated type objects and matched
    by name). The supported route is the same one student_130 uses: publish the SDK ROS message and
    let igris_c_bridge forward ROS -> native. Interface mirrors LowCmdPublisher.send exactly.
    """

    def __init__(self, node, topic: str = TOPIC):
        from rclpy.qos import qos_profile_sensor_data
        from igris_c_sdk.msg import LowCmd as RosLowCmd
        self._RosLowCmd = RosLowCmd
        self._pub = node.create_publisher(RosLowCmd, topic, qos_profile_sensor_data)
        self._seq = 0

    def send(self, q31, dq31, kp31, kd31, kinematic_modes: "KinematicMode" = PJS):
        assert len(q31) == len(dq31) == len(kp31) == len(kd31) == NUM_MOTORS
        m = self._RosLowCmd()
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        m.header.seq = self._seq
        m.header.frame_id = ""  # raw path의 _FRAME_ID는 char[256] 배열이라 ROS string에 부적합
        km = int(getattr(kinematic_modes, "value", kinematic_modes))
        m.kinematic_modes = [km] * 5
        for i in range(NUM_MOTORS):
            mc = m.motors[i]
            mc.id = i
            mc.q = float(q31[i]); mc.dq = float(dq31[i]); mc.tau = 0.0
            mc.kp = float(kp31[i]); mc.kd = float(kd31[i])
        self._pub.publish(m)


# --- LowState (igris_c_001/rt/lowstate) — optional; lets the spike self-verify
#     that the controller applied our lowcmd (joints track the commanded pose). --
@dataclass
@annotate.final
@annotate.autoid("sequential")
class IMUState(idl.IdlStruct, typename="igris_c.msg.dds.IMUState"):
    quaternion: types.array[types.float32, 4]
    gyroscope: types.array[types.float32, 3]
    accelerometer: types.array[types.float32, 3]
    rpy: types.array[types.float32, 3]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState(idl.IdlStruct, typename="igris_c.msg.dds.MotorState"):
    q: types.float32
    dq: types.float32
    tau_est: types.float32
    temperature: types.int16
    status_bits: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class JointState(idl.IdlStruct, typename="igris_c.msg.dds.JointState"):
    q: types.float32
    dq: types.float32
    tau_est: types.float32
    status_bits: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowState(idl.IdlStruct, typename="igris_c.msg.dds.LowState"):
    header: Header
    imu_state: IMUState
    motor_state: types.array[MotorState, 31]
    joint_state: types.array[JointState, 31]


class LowStateSub:
    """Best-effort cyclonedds reader for igris_c_001/rt/lowstate."""

    def __init__(self, domain: int = 0):
        self.dp = DomainParticipant(domain)
        qos = Qos(Policy.Reliability.BestEffort,
                  Policy.Durability.Volatile,
                  Policy.History.KeepLast(5))
        self.dr = DataReader(Subscriber(self.dp),
                             Topic(self.dp, "igris_c_001/rt/lowstate", LowState, qos=qos),
                             qos=qos)

    def joint_q(self):
        """Latest 31 measured joint positions, or None if no sample yet."""
        latest = None
        for s in self.dr.take(N=20):
            latest = s
        if latest is None:
            return None
        return [float(latest.joint_state[i].q) for i in range(NUM_MOTORS)]
