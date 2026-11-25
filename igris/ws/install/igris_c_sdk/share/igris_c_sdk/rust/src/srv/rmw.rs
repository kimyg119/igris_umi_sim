#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__BmsInitCmd_Request() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__BmsInitCmd_Request__init(msg: *mut BmsInitCmd_Request) -> bool;
    fn igris_c_sdk__srv__BmsInitCmd_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BmsInitCmd_Request>, size: usize) -> bool;
    fn igris_c_sdk__srv__BmsInitCmd_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BmsInitCmd_Request>);
    fn igris_c_sdk__srv__BmsInitCmd_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BmsInitCmd_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<BmsInitCmd_Request>) -> bool;
}

// Corresponds to igris_c_sdk__srv__BmsInitCmd_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BmsInitCmd_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__BmsInitCmd_Request__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__BmsInitCmd_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BmsInitCmd_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__BmsInitCmd_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__BmsInitCmd_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__BmsInitCmd_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BmsInitCmd_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BmsInitCmd_Request where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/BmsInitCmd_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__BmsInitCmd_Request() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__BmsInitCmd_Response() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__BmsInitCmd_Response__init(msg: *mut BmsInitCmd_Response) -> bool;
    fn igris_c_sdk__srv__BmsInitCmd_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BmsInitCmd_Response>, size: usize) -> bool;
    fn igris_c_sdk__srv__BmsInitCmd_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BmsInitCmd_Response>);
    fn igris_c_sdk__srv__BmsInitCmd_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BmsInitCmd_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<BmsInitCmd_Response>) -> bool;
}

// Corresponds to igris_c_sdk__srv__BmsInitCmd_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BmsInitCmd_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub error_code: i32,

}



impl Default for BmsInitCmd_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__BmsInitCmd_Response__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__BmsInitCmd_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BmsInitCmd_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__BmsInitCmd_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__BmsInitCmd_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__BmsInitCmd_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BmsInitCmd_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BmsInitCmd_Response where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/BmsInitCmd_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__BmsInitCmd_Response() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__TorqueCmd_Request() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__TorqueCmd_Request__init(msg: *mut TorqueCmd_Request) -> bool;
    fn igris_c_sdk__srv__TorqueCmd_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<TorqueCmd_Request>, size: usize) -> bool;
    fn igris_c_sdk__srv__TorqueCmd_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<TorqueCmd_Request>);
    fn igris_c_sdk__srv__TorqueCmd_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<TorqueCmd_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<TorqueCmd_Request>) -> bool;
}

// Corresponds to igris_c_sdk__srv__TorqueCmd_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TorqueCmd_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__TorqueCmd_Request__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__TorqueCmd_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for TorqueCmd_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__TorqueCmd_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__TorqueCmd_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__TorqueCmd_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for TorqueCmd_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for TorqueCmd_Request where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/TorqueCmd_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__TorqueCmd_Request() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__TorqueCmd_Response() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__TorqueCmd_Response__init(msg: *mut TorqueCmd_Response) -> bool;
    fn igris_c_sdk__srv__TorqueCmd_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<TorqueCmd_Response>, size: usize) -> bool;
    fn igris_c_sdk__srv__TorqueCmd_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<TorqueCmd_Response>);
    fn igris_c_sdk__srv__TorqueCmd_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<TorqueCmd_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<TorqueCmd_Response>) -> bool;
}

// Corresponds to igris_c_sdk__srv__TorqueCmd_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TorqueCmd_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub error_code: i32,

}



impl Default for TorqueCmd_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__TorqueCmd_Response__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__TorqueCmd_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for TorqueCmd_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__TorqueCmd_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__TorqueCmd_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__TorqueCmd_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for TorqueCmd_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for TorqueCmd_Response where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/TorqueCmd_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__TorqueCmd_Response() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__ControlModeCommandRequest_Request() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__ControlModeCommandRequest_Request__init(msg: *mut ControlModeCommandRequest_Request) -> bool;
    fn igris_c_sdk__srv__ControlModeCommandRequest_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Request>, size: usize) -> bool;
    fn igris_c_sdk__srv__ControlModeCommandRequest_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Request>);
    fn igris_c_sdk__srv__ControlModeCommandRequest_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Request>) -> bool;
}

// Corresponds to igris_c_sdk__srv__ControlModeCommandRequest_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ControlModeCommandRequest_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub command_type: u8,


    // This member is not documented.
    #[allow(missing_docs)]
    pub preset_id: rosidl_runtime_rs::String,


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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__ControlModeCommandRequest_Request__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__ControlModeCommandRequest_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ControlModeCommandRequest_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__ControlModeCommandRequest_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__ControlModeCommandRequest_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__ControlModeCommandRequest_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ControlModeCommandRequest_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ControlModeCommandRequest_Request where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/ControlModeCommandRequest_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__ControlModeCommandRequest_Request() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__ControlModeCommandRequest_Response() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__ControlModeCommandRequest_Response__init(msg: *mut ControlModeCommandRequest_Response) -> bool;
    fn igris_c_sdk__srv__ControlModeCommandRequest_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Response>, size: usize) -> bool;
    fn igris_c_sdk__srv__ControlModeCommandRequest_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Response>);
    fn igris_c_sdk__srv__ControlModeCommandRequest_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<ControlModeCommandRequest_Response>) -> bool;
}

// Corresponds to igris_c_sdk__srv__ControlModeCommandRequest_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ControlModeCommandRequest_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub error_code: i32,

}



impl Default for ControlModeCommandRequest_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__ControlModeCommandRequest_Response__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__ControlModeCommandRequest_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ControlModeCommandRequest_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__ControlModeCommandRequest_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__ControlModeCommandRequest_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__ControlModeCommandRequest_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ControlModeCommandRequest_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ControlModeCommandRequest_Response where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/ControlModeCommandRequest_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__ControlModeCommandRequest_Response() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__HandInitRequest_Request() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__HandInitRequest_Request__init(msg: *mut HandInitRequest_Request) -> bool;
    fn igris_c_sdk__srv__HandInitRequest_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<HandInitRequest_Request>, size: usize) -> bool;
    fn igris_c_sdk__srv__HandInitRequest_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<HandInitRequest_Request>);
    fn igris_c_sdk__srv__HandInitRequest_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<HandInitRequest_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<HandInitRequest_Request>) -> bool;
}

// Corresponds to igris_c_sdk__srv__HandInitRequest_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HandInitRequest_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,

}



impl Default for HandInitRequest_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__HandInitRequest_Request__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__HandInitRequest_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for HandInitRequest_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__HandInitRequest_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__HandInitRequest_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__HandInitRequest_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for HandInitRequest_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for HandInitRequest_Request where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/HandInitRequest_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__HandInitRequest_Request() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__HandInitRequest_Response() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__HandInitRequest_Response__init(msg: *mut HandInitRequest_Response) -> bool;
    fn igris_c_sdk__srv__HandInitRequest_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<HandInitRequest_Response>, size: usize) -> bool;
    fn igris_c_sdk__srv__HandInitRequest_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<HandInitRequest_Response>);
    fn igris_c_sdk__srv__HandInitRequest_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<HandInitRequest_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<HandInitRequest_Response>) -> bool;
}

// Corresponds to igris_c_sdk__srv__HandInitRequest_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct HandInitRequest_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub error_code: i32,

}



impl Default for HandInitRequest_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__HandInitRequest_Response__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__HandInitRequest_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for HandInitRequest_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__HandInitRequest_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__HandInitRequest_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__HandInitRequest_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for HandInitRequest_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for HandInitRequest_Response where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/HandInitRequest_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__HandInitRequest_Response() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__MujocoSimCmd_Request() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__MujocoSimCmd_Request__init(msg: *mut MujocoSimCmd_Request) -> bool;
    fn igris_c_sdk__srv__MujocoSimCmd_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<MujocoSimCmd_Request>, size: usize) -> bool;
    fn igris_c_sdk__srv__MujocoSimCmd_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<MujocoSimCmd_Request>);
    fn igris_c_sdk__srv__MujocoSimCmd_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<MujocoSimCmd_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<MujocoSimCmd_Request>) -> bool;
}

// Corresponds to igris_c_sdk__srv__MujocoSimCmd_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MujocoSimCmd_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__MujocoSimCmd_Request__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__MujocoSimCmd_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for MujocoSimCmd_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__MujocoSimCmd_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__MujocoSimCmd_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__MujocoSimCmd_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for MujocoSimCmd_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for MujocoSimCmd_Request where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/MujocoSimCmd_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__MujocoSimCmd_Request() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__MujocoSimCmd_Response() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__srv__MujocoSimCmd_Response__init(msg: *mut MujocoSimCmd_Response) -> bool;
    fn igris_c_sdk__srv__MujocoSimCmd_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<MujocoSimCmd_Response>, size: usize) -> bool;
    fn igris_c_sdk__srv__MujocoSimCmd_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<MujocoSimCmd_Response>);
    fn igris_c_sdk__srv__MujocoSimCmd_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<MujocoSimCmd_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<MujocoSimCmd_Response>) -> bool;
}

// Corresponds to igris_c_sdk__srv__MujocoSimCmd_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MujocoSimCmd_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub request_id: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub error_code: i32,

}



impl Default for MujocoSimCmd_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__srv__MujocoSimCmd_Response__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__srv__MujocoSimCmd_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for MujocoSimCmd_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__MujocoSimCmd_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__MujocoSimCmd_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__srv__MujocoSimCmd_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for MujocoSimCmd_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for MujocoSimCmd_Response where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/srv/MujocoSimCmd_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__srv__MujocoSimCmd_Response() }
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


