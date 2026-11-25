#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__Header() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__Header__init(msg: *mut Header) -> bool;
    fn igris_c_sdk__msg__Header__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Header>, size: usize) -> bool;
    fn igris_c_sdk__msg__Header__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Header>);
    fn igris_c_sdk__msg__Header__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Header>, out_seq: *mut rosidl_runtime_rs::Sequence<Header>) -> bool;
}

// Corresponds to igris_c_sdk__msg__Header
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
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
    pub frame_id: rosidl_runtime_rs::String,

}



impl Default for Header {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__Header__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__Header__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Header {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__Header__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__Header__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__Header__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Header {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Header where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/Header";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__Header() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__IMUState() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__IMUState__init(msg: *mut IMUState) -> bool;
    fn igris_c_sdk__msg__IMUState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<IMUState>, size: usize) -> bool;
    fn igris_c_sdk__msg__IMUState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<IMUState>);
    fn igris_c_sdk__msg__IMUState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<IMUState>, out_seq: *mut rosidl_runtime_rs::Sequence<IMUState>) -> bool;
}

// Corresponds to igris_c_sdk__msg__IMUState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__IMUState__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__IMUState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for IMUState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__IMUState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__IMUState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__IMUState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for IMUState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for IMUState where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/IMUState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__IMUState() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__MotorCmd() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__MotorCmd__init(msg: *mut MotorCmd) -> bool;
    fn igris_c_sdk__msg__MotorCmd__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<MotorCmd>, size: usize) -> bool;
    fn igris_c_sdk__msg__MotorCmd__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<MotorCmd>);
    fn igris_c_sdk__msg__MotorCmd__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<MotorCmd>, out_seq: *mut rosidl_runtime_rs::Sequence<MotorCmd>) -> bool;
}

// Corresponds to igris_c_sdk__msg__MotorCmd
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__MotorCmd__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__MotorCmd__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for MotorCmd {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__MotorCmd__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__MotorCmd__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__MotorCmd__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for MotorCmd {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for MotorCmd where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/MotorCmd";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__MotorCmd() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__MotorState() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__MotorState__init(msg: *mut MotorState) -> bool;
    fn igris_c_sdk__msg__MotorState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<MotorState>, size: usize) -> bool;
    fn igris_c_sdk__msg__MotorState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<MotorState>);
    fn igris_c_sdk__msg__MotorState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<MotorState>, out_seq: *mut rosidl_runtime_rs::Sequence<MotorState>) -> bool;
}

// Corresponds to igris_c_sdk__msg__MotorState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__MotorState__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__MotorState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for MotorState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__MotorState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__MotorState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__MotorState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for MotorState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for MotorState where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/MotorState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__MotorState() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__JointState() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__JointState__init(msg: *mut JointState) -> bool;
    fn igris_c_sdk__msg__JointState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<JointState>, size: usize) -> bool;
    fn igris_c_sdk__msg__JointState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<JointState>);
    fn igris_c_sdk__msg__JointState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<JointState>, out_seq: *mut rosidl_runtime_rs::Sequence<JointState>) -> bool;
}

// Corresponds to igris_c_sdk__msg__JointState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__JointState__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__JointState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for JointState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__JointState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__JointState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__JointState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for JointState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for JointState where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/JointState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__JointState() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__LowCmd() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__LowCmd__init(msg: *mut LowCmd) -> bool;
    fn igris_c_sdk__msg__LowCmd__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<LowCmd>, size: usize) -> bool;
    fn igris_c_sdk__msg__LowCmd__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<LowCmd>);
    fn igris_c_sdk__msg__LowCmd__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<LowCmd>, out_seq: *mut rosidl_runtime_rs::Sequence<LowCmd>) -> bool;
}

// Corresponds to igris_c_sdk__msg__LowCmd
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LowCmd {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub kinematic_modes: [u8; 5],


    // This member is not documented.
    #[allow(missing_docs)]
    pub motors: [super::super::msg::rmw::MotorCmd; 31],

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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__LowCmd__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__LowCmd__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for LowCmd {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__LowCmd__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__LowCmd__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__LowCmd__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for LowCmd {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for LowCmd where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/LowCmd";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__LowCmd() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__LowState() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__LowState__init(msg: *mut LowState) -> bool;
    fn igris_c_sdk__msg__LowState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<LowState>, size: usize) -> bool;
    fn igris_c_sdk__msg__LowState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<LowState>);
    fn igris_c_sdk__msg__LowState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<LowState>, out_seq: *mut rosidl_runtime_rs::Sequence<LowState>) -> bool;
}

// Corresponds to igris_c_sdk__msg__LowState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LowState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub imu_state: super::super::msg::rmw::IMUState,


    // This member is not documented.
    #[allow(missing_docs)]
    pub motor_state: [super::super::msg::rmw::MotorState; 31],


    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_state: [super::super::msg::rmw::JointState; 31],

}



impl Default for LowState {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__LowState__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__LowState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for LowState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__LowState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__LowState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__LowState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for LowState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for LowState where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/LowState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__LowState() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__BmsState() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__BmsState__init(msg: *mut BmsState) -> bool;
    fn igris_c_sdk__msg__BmsState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<BmsState>, size: usize) -> bool;
    fn igris_c_sdk__msg__BmsState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<BmsState>);
    fn igris_c_sdk__msg__BmsState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<BmsState>, out_seq: *mut rosidl_runtime_rs::Sequence<BmsState>) -> bool;
}

// Corresponds to igris_c_sdk__msg__BmsState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BmsState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__BmsState__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__BmsState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for BmsState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__BmsState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__BmsState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__BmsState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for BmsState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for BmsState where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/BmsState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__BmsState() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__RobotState() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__RobotState__init(msg: *mut RobotState) -> bool;
    fn igris_c_sdk__msg__RobotState__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<RobotState>, size: usize) -> bool;
    fn igris_c_sdk__msg__RobotState__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<RobotState>);
    fn igris_c_sdk__msg__RobotState__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<RobotState>, out_seq: *mut rosidl_runtime_rs::Sequence<RobotState>) -> bool;
}

// Corresponds to igris_c_sdk__msg__RobotState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RobotState {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__RobotState__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__RobotState__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for RobotState {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__RobotState__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__RobotState__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__RobotState__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for RobotState {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for RobotState where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/RobotState";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__RobotState() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__ServiceResponse() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__ServiceResponse__init(msg: *mut ServiceResponse) -> bool;
    fn igris_c_sdk__msg__ServiceResponse__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ServiceResponse>, size: usize) -> bool;
    fn igris_c_sdk__msg__ServiceResponse__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ServiceResponse>);
    fn igris_c_sdk__msg__ServiceResponse__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ServiceResponse>, out_seq: *mut rosidl_runtime_rs::Sequence<ServiceResponse>) -> bool;
}

// Corresponds to igris_c_sdk__msg__ServiceResponse
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ServiceResponse {

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



impl Default for ServiceResponse {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__ServiceResponse__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__ServiceResponse__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ServiceResponse {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__ServiceResponse__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__ServiceResponse__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__ServiceResponse__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ServiceResponse {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ServiceResponse where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/ServiceResponse";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__ServiceResponse() }
  }
}


#[link(name = "igris_c_sdk__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__WalkTwist() -> *const std::ffi::c_void;
}

#[link(name = "igris_c_sdk__rosidl_generator_c")]
extern "C" {
    fn igris_c_sdk__msg__WalkTwist__init(msg: *mut WalkTwist) -> bool;
    fn igris_c_sdk__msg__WalkTwist__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<WalkTwist>, size: usize) -> bool;
    fn igris_c_sdk__msg__WalkTwist__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<WalkTwist>);
    fn igris_c_sdk__msg__WalkTwist__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<WalkTwist>, out_seq: *mut rosidl_runtime_rs::Sequence<WalkTwist>) -> bool;
}

// Corresponds to igris_c_sdk__msg__WalkTwist
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct WalkTwist {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: super::super::msg::rmw::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: geometry_msgs::msg::rmw::Twist,

}



impl Default for WalkTwist {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !igris_c_sdk__msg__WalkTwist__init(&mut msg as *mut _) {
        panic!("Call to igris_c_sdk__msg__WalkTwist__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for WalkTwist {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__WalkTwist__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__WalkTwist__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { igris_c_sdk__msg__WalkTwist__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for WalkTwist {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for WalkTwist where Self: Sized {
  const TYPE_NAME: &'static str = "igris_c_sdk/msg/WalkTwist";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__igris_c_sdk__msg__WalkTwist() }
  }
}


