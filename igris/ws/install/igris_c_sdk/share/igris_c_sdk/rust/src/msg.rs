#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to igris_c_sdk__msg__Header

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Header {

    // This member is not documented.
    #[allow(missing_docs)]
    pub seq: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub sec: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub nanosec: u32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub frame_id: std::string::String,

}



impl Default for Header {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Header::default())
  }
}

impl rosidl_runtime_rs::Message for Header {
  type RmwMsg = super::msg::rmw::Header;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        seq: msg.seq,
        sec: msg.sec,
        nanosec: msg.nanosec,
        frame_id: msg.frame_id.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      seq: msg.seq,
      sec: msg.sec,
      nanosec: msg.nanosec,
        frame_id: msg.frame_id.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      seq: msg.seq,
      sec: msg.sec,
      nanosec: msg.nanosec,
      frame_id: msg.frame_id.to_string(),
    }
  }
}


// Corresponds to igris_c_sdk__msg__IMUState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct IMUState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub quaternion: [f32; 4],


    // This member is not documented.
    #[allow(missing_docs)]
    pub gyroscope: [f32; 3],


    // This member is not documented.
    #[allow(missing_docs)]
    pub accelerometer: [f32; 3],


    // This member is not documented.
    #[allow(missing_docs)]
    pub rpy: [f32; 3],

}



impl Default for IMUState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::IMUState::default())
  }
}

impl rosidl_runtime_rs::Message for IMUState {
  type RmwMsg = super::msg::rmw::IMUState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        quaternion: msg.quaternion,
        gyroscope: msg.gyroscope,
        accelerometer: msg.accelerometer,
        rpy: msg.rpy,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        quaternion: msg.quaternion,
        gyroscope: msg.gyroscope,
        accelerometer: msg.accelerometer,
        rpy: msg.rpy,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      quaternion: msg.quaternion,
      gyroscope: msg.gyroscope,
      accelerometer: msg.accelerometer,
      rpy: msg.rpy,
    }
  }
}


// Corresponds to igris_c_sdk__msg__MotorCmd

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MotorCmd {

    // This member is not documented.
    #[allow(missing_docs)]
    pub id: u16,


    // This member is not documented.
    #[allow(missing_docs)]
    pub q: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub dq: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub tau: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub kp: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub kd: f32,

}



impl Default for MotorCmd {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::MotorCmd::default())
  }
}

impl rosidl_runtime_rs::Message for MotorCmd {
  type RmwMsg = super::msg::rmw::MotorCmd;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        id: msg.id,
        q: msg.q,
        dq: msg.dq,
        tau: msg.tau,
        kp: msg.kp,
        kd: msg.kd,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      id: msg.id,
      q: msg.q,
      dq: msg.dq,
      tau: msg.tau,
      kp: msg.kp,
      kd: msg.kd,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      id: msg.id,
      q: msg.q,
      dq: msg.dq,
      tau: msg.tau,
      kp: msg.kp,
      kd: msg.kd,
    }
  }
}


// Corresponds to igris_c_sdk__msg__MotorState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MotorState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub q: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub dq: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub tau_est: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub temperature: i16,


    // This member is not documented.
    #[allow(missing_docs)]
    pub status_bits: u32,

}



impl Default for MotorState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::MotorState::default())
  }
}

impl rosidl_runtime_rs::Message for MotorState {
  type RmwMsg = super::msg::rmw::MotorState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        q: msg.q,
        dq: msg.dq,
        tau_est: msg.tau_est,
        temperature: msg.temperature,
        status_bits: msg.status_bits,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      q: msg.q,
      dq: msg.dq,
      tau_est: msg.tau_est,
      temperature: msg.temperature,
      status_bits: msg.status_bits,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      q: msg.q,
      dq: msg.dq,
      tau_est: msg.tau_est,
      temperature: msg.temperature,
      status_bits: msg.status_bits,
    }
  }
}


// Corresponds to igris_c_sdk__msg__JointState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub q: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub dq: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub tau_est: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub status_bits: u32,

}



impl Default for JointState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::JointState::default())
  }
}

impl rosidl_runtime_rs::Message for JointState {
  type RmwMsg = super::msg::rmw::JointState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        q: msg.q,
        dq: msg.dq,
        tau_est: msg.tau_est,
        status_bits: msg.status_bits,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      q: msg.q,
      dq: msg.dq,
      tau_est: msg.tau_est,
      status_bits: msg.status_bits,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      q: msg.q,
      dq: msg.dq,
      tau_est: msg.tau_est,
      status_bits: msg.status_bits,
    }
  }
}


// Corresponds to igris_c_sdk__msg__LowCmd

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LowCmd {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub kinematic_modes: [u8; 5],


    // This member is not documented.
    #[allow(missing_docs)]
    pub motors: [super::msg::MotorCmd; 31],

}

impl LowCmd {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const KINEMATIC_MODE_MS: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const KINEMATIC_MODE_PJS: u8 = 1;

}


impl Default for LowCmd {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::LowCmd::default())
  }
}

impl rosidl_runtime_rs::Message for LowCmd {
  type RmwMsg = super::msg::rmw::LowCmd;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        kinematic_modes: msg.kinematic_modes,
        motors: msg.motors
          .map(|elem| super::msg::MotorCmd::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned()),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        kinematic_modes: msg.kinematic_modes,
        motors: msg.motors
          .iter()
          .map(|elem| super::msg::MotorCmd::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect::<Vec<_>>()
          .try_into()
          .unwrap(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      kinematic_modes: msg.kinematic_modes,
      motors: msg.motors
        .map(super::msg::MotorCmd::from_rmw_message),
    }
  }
}


// Corresponds to igris_c_sdk__msg__LowState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LowState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub imu_state: super::msg::IMUState,


    // This member is not documented.
    #[allow(missing_docs)]
    pub motor_state: [super::msg::MotorState; 31],


    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_state: [super::msg::JointState; 31],

}



impl Default for LowState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::LowState::default())
  }
}

impl rosidl_runtime_rs::Message for LowState {
  type RmwMsg = super::msg::rmw::LowState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        imu_state: super::msg::IMUState::into_rmw_message(std::borrow::Cow::Owned(msg.imu_state)).into_owned(),
        motor_state: msg.motor_state
          .map(|elem| super::msg::MotorState::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned()),
        joint_state: msg.joint_state
          .map(|elem| super::msg::JointState::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned()),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        imu_state: super::msg::IMUState::into_rmw_message(std::borrow::Cow::Borrowed(&msg.imu_state)).into_owned(),
        motor_state: msg.motor_state
          .iter()
          .map(|elem| super::msg::MotorState::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect::<Vec<_>>()
          .try_into()
          .unwrap(),
        joint_state: msg.joint_state
          .iter()
          .map(|elem| super::msg::JointState::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect::<Vec<_>>()
          .try_into()
          .unwrap(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      imu_state: super::msg::IMUState::from_rmw_message(msg.imu_state),
      motor_state: msg.motor_state
        .map(super::msg::MotorState::from_rmw_message),
      joint_state: msg.joint_state
        .map(super::msg::JointState::from_rmw_message),
    }
  }
}


// Corresponds to igris_c_sdk__msg__BmsState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BmsState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub body_power: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub legs_power: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub estop: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub connect: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub battery: f32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub bms_init_state: u8,

}

impl BmsState {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const RELAY_OFF: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const RELAY_ON: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ESTOP_RELEASED: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ESTOP_PRESSED: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const BMS_DISCONNECTED: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const BMS_CONNECTED: u8 = 1;

    /// 0b00: neither (bit0=BMS, bit1=motor)
    pub const BMS_NOT_INITIALIZED: u8 = 0;

    /// 0b01: only BMS initialized
    pub const BMS_INITIALIZED: u8 = 1;

    /// 0b10: only motor initialized
    pub const MOTOR_INITIALIZED: u8 = 2;

    /// 0b11: BMS | motor
    pub const BOTH_INITIALIZED: u8 = 3;

}


impl Default for BmsState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::BmsState::default())
  }
}

impl rosidl_runtime_rs::Message for BmsState {
  type RmwMsg = super::msg::rmw::BmsState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        body_power: msg.body_power,
        legs_power: msg.legs_power,
        estop: msg.estop,
        connect: msg.connect,
        battery: msg.battery,
        bms_init_state: msg.bms_init_state,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      body_power: msg.body_power,
      legs_power: msg.legs_power,
      estop: msg.estop,
      connect: msg.connect,
      battery: msg.battery,
      bms_init_state: msg.bms_init_state,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      body_power: msg.body_power,
      legs_power: msg.legs_power,
      estop: msg.estop,
      connect: msg.connect,
      battery: msg.battery,
      bms_init_state: msg.bms_init_state,
    }
  }
}


// Corresponds to igris_c_sdk__msg__RobotState

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RobotState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub state: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub environment: u8,

}

impl RobotState {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_NOT_READY: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_IDLE: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_STOP: u8 = 2;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_LOW: u8 = 3;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_HIGH_MOTION_ACTIVE: u8 = 4;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_HIGH_MOTION_STANDBY: u8 = 5;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_WALK_LOW: u8 = 6;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_WALK_HIGH_MOTION_ACTIVE: u8 = 7;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_WALK_HIGH_MOTION_STANDBY: u8 = 8;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_STATE_WHOLE_BODY_CONTROL: u8 = 9;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_ENV_REAL: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const ROBOT_ENV_SIM: u8 = 1;

}


impl Default for RobotState {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::RobotState::default())
  }
}

impl rosidl_runtime_rs::Message for RobotState {
  type RmwMsg = super::msg::rmw::RobotState;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        state: msg.state,
        environment: msg.environment,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      state: msg.state,
      environment: msg.environment,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      state: msg.state,
      environment: msg.environment,
    }
  }
}


// Corresponds to igris_c_sdk__msg__ServiceResponse

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ServiceResponse {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub error_code: i32,

}



impl Default for ServiceResponse {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::ServiceResponse::default())
  }
}

impl rosidl_runtime_rs::Message for ServiceResponse {
  type RmwMsg = super::msg::rmw::ServiceResponse;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
        success: msg.success,
        message: msg.message.as_str().into(),
        error_code: msg.error_code,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      success: msg.success,
        message: msg.message.as_str().into(),
      error_code: msg.error_code,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      request_id: msg.request_id.to_string(),
      success: msg.success,
      message: msg.message.to_string(),
      error_code: msg.error_code,
    }
  }
}


// Corresponds to igris_c_sdk__msg__WalkTwist

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct WalkTwist {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: geometry_msgs::msg::Twist,

}



impl Default for WalkTwist {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::WalkTwist::default())
  }
}

impl rosidl_runtime_rs::Message for WalkTwist {
  type RmwMsg = super::msg::rmw::WalkTwist;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        twist: geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(msg.twist)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        twist: geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(&msg.twist)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      twist: geometry_msgs::msg::Twist::from_rmw_message(msg.twist),
    }
  }
}


