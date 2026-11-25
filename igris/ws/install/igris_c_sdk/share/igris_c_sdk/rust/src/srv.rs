#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};




// Corresponds to igris_c_sdk__srv__BmsInitCmd_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BmsInitCmd_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub init: u8,

}

impl BmsInitCmd_Request {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const BMS_INIT_NONE: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const BMS_INIT: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const MOTOR_INIT: u8 = 2;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const BMS_AND_MOTOR_INIT: u8 = 3;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const BMS_OFF: u8 = 4;

}


impl Default for BmsInitCmd_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::BmsInitCmd_Request::default())
  }
}

impl rosidl_runtime_rs::Message for BmsInitCmd_Request {
  type RmwMsg = super::srv::rmw::BmsInitCmd_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
        init: msg.init,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      init: msg.init,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      request_id: msg.request_id.to_string(),
      init: msg.init,
    }
  }
}


// Corresponds to igris_c_sdk__srv__BmsInitCmd_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BmsInitCmd_Response {

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



impl Default for BmsInitCmd_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::BmsInitCmd_Response::default())
  }
}

impl rosidl_runtime_rs::Message for BmsInitCmd_Response {
  type RmwMsg = super::srv::rmw::BmsInitCmd_Response;

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


// Corresponds to igris_c_sdk__srv__TorqueCmd_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TorqueCmd_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub torque: u8,

}

impl TorqueCmd_Request {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TORQUE_NONE: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TORQUE_ON: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TORQUE_OFF: u8 = 2;

}


impl Default for TorqueCmd_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::TorqueCmd_Request::default())
  }
}

impl rosidl_runtime_rs::Message for TorqueCmd_Request {
  type RmwMsg = super::srv::rmw::TorqueCmd_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
        torque: msg.torque,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      torque: msg.torque,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      request_id: msg.request_id.to_string(),
      torque: msg.torque,
    }
  }
}


// Corresponds to igris_c_sdk__srv__TorqueCmd_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TorqueCmd_Response {

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



impl Default for TorqueCmd_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::TorqueCmd_Response::default())
  }
}

impl rosidl_runtime_rs::Message for TorqueCmd_Response {
  type RmwMsg = super::srv::rmw::TorqueCmd_Response;

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


// Corresponds to igris_c_sdk__srv__ControlModeCommandRequest_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ControlModeCommandRequest_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub command_type: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub preset_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub is_cyclic: bool,

}

impl ControlModeCommandRequest_Request {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_MOTION_PRESET: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_MOTION_PRESET_CYCLIC_TOGGLE: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_JOINT_POSITION_HOLD: u8 = 2;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_MOTION_STOP: u8 = 3;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_WALKMODE_ON: u8 = 4;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL: u8 = 5;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_LOW_LEVEL_WALKMODE_ON: u8 = 6;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_HIGH_LEVEL_WALKMODE_ON: u8 = 7;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_HIGH_LEVEL_JOINT_CONTROL: u8 = 8;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_1: u8 = 9;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_2: u8 = 10;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_3: u8 = 11;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_4: u8 = 12;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_5: u8 = 13;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_6: u8 = 14;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_7: u8 = 15;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_8: u8 = 16;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_9: u8 = 17;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const CONTROL_MODE_CMD_CUSTOM_MODE_10: u8 = 18;

}


impl Default for ControlModeCommandRequest_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::ControlModeCommandRequest_Request::default())
  }
}

impl rosidl_runtime_rs::Message for ControlModeCommandRequest_Request {
  type RmwMsg = super::srv::rmw::ControlModeCommandRequest_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
        command_type: msg.command_type,
        preset_id: msg.preset_id.as_str().into(),
        is_cyclic: msg.is_cyclic,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      command_type: msg.command_type,
        preset_id: msg.preset_id.as_str().into(),
      is_cyclic: msg.is_cyclic,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      request_id: msg.request_id.to_string(),
      command_type: msg.command_type,
      preset_id: msg.preset_id.to_string(),
      is_cyclic: msg.is_cyclic,
    }
  }
}


// Corresponds to igris_c_sdk__srv__ControlModeCommandRequest_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ControlModeCommandRequest_Response {

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



impl Default for ControlModeCommandRequest_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::ControlModeCommandRequest_Response::default())
  }
}

impl rosidl_runtime_rs::Message for ControlModeCommandRequest_Response {
  type RmwMsg = super::srv::rmw::ControlModeCommandRequest_Response;

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


// Corresponds to igris_c_sdk__srv__HandInitRequest_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HandInitRequest_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: std::string::String,

}



impl Default for HandInitRequest_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::HandInitRequest_Request::default())
  }
}

impl rosidl_runtime_rs::Message for HandInitRequest_Request {
  type RmwMsg = super::srv::rmw::HandInitRequest_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      request_id: msg.request_id.to_string(),
    }
  }
}


// Corresponds to igris_c_sdk__srv__HandInitRequest_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HandInitRequest_Response {

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



impl Default for HandInitRequest_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::HandInitRequest_Response::default())
  }
}

impl rosidl_runtime_rs::Message for HandInitRequest_Response {
  type RmwMsg = super::srv::rmw::HandInitRequest_Response;

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


// Corresponds to igris_c_sdk__srv__MujocoSimCmd_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MujocoSimCmd_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub command_type: u8,

}

impl MujocoSimCmd_Request {

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const MUJOCO_SIM_PAUSE: u8 = 0;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const MUJOCO_SIM_RESUME: u8 = 1;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const MUJOCO_SIM_RELOAD: u8 = 2;


    // This constant is not documented.
    #[allow(missing_docs)]
    pub const MUJOCO_SIM_RESET: u8 = 3;

}


impl Default for MujocoSimCmd_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::MujocoSimCmd_Request::default())
  }
}

impl rosidl_runtime_rs::Message for MujocoSimCmd_Request {
  type RmwMsg = super::srv::rmw::MujocoSimCmd_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
        command_type: msg.command_type,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: super::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        request_id: msg.request_id.as_str().into(),
      command_type: msg.command_type,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: super::msg::Header::from_rmw_message(msg.header),
      request_id: msg.request_id.to_string(),
      command_type: msg.command_type,
    }
  }
}


// Corresponds to igris_c_sdk__srv__MujocoSimCmd_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MujocoSimCmd_Response {

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



impl Default for MujocoSimCmd_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::MujocoSimCmd_Response::default())
  }
}

impl rosidl_runtime_rs::Message for MujocoSimCmd_Response {
  type RmwMsg = super::srv::rmw::MujocoSimCmd_Response;

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






#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__BmsInitCmd() -> *const std::ffi::c_void;
}

// Corresponds to igris_c_sdk__srv__BmsInitCmd
#[allow(missing_docs, non_camel_case_types)]
pub struct BmsInitCmd;

impl rosidl_runtime_rs::Service for BmsInitCmd {
    type Request = BmsInitCmd_Request;
    type Response = BmsInitCmd_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__BmsInitCmd() }
    }
}




#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__TorqueCmd() -> *const std::ffi::c_void;
}

// Corresponds to igris_c_sdk__srv__TorqueCmd
#[allow(missing_docs, non_camel_case_types)]
pub struct TorqueCmd;

impl rosidl_runtime_rs::Service for TorqueCmd {
    type Request = TorqueCmd_Request;
    type Response = TorqueCmd_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__TorqueCmd() }
    }
}




#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__ControlModeCommandRequest() -> *const std::ffi::c_void;
}

// Corresponds to igris_c_sdk__srv__ControlModeCommandRequest
#[allow(missing_docs, non_camel_case_types)]
pub struct ControlModeCommandRequest;

impl rosidl_runtime_rs::Service for ControlModeCommandRequest {
    type Request = ControlModeCommandRequest_Request;
    type Response = ControlModeCommandRequest_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__ControlModeCommandRequest() }
    }
}




#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__HandInitRequest() -> *const std::ffi::c_void;
}

// Corresponds to igris_c_sdk__srv__HandInitRequest
#[allow(missing_docs, non_camel_case_types)]
pub struct HandInitRequest;

impl rosidl_runtime_rs::Service for HandInitRequest {
    type Request = HandInitRequest_Request;
    type Response = HandInitRequest_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__HandInitRequest() }
    }
}




#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__MujocoSimCmd() -> *const std::ffi::c_void;
}

// Corresponds to igris_c_sdk__srv__MujocoSimCmd
#[allow(missing_docs, non_camel_case_types)]
pub struct MujocoSimCmd;

impl rosidl_runtime_rs::Service for MujocoSimCmd {
    type Request = MujocoSimCmd_Request;
    type Response = MujocoSimCmd_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__igris_c_sdk__srv__MujocoSimCmd() }
    }
}


