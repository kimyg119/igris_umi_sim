#!/usr/bin/env python3

import queue
import threading
import time
import tkinter as tk
from typing import Callable, List, Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

HAND_MOTOR_IDS = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]


class HandBridgeNode(Node):
    def __init__(self, status_queue: queue.Queue, position_queue: queue.Queue) -> None:
        super().__init__("hand_control_gui")
        self._status_queue = status_queue
        self._position_queue = position_queue
        self._cmd_pub = self.create_publisher(String, "igris_c/hand/cmd", 10)
        self._targets_pub = self.create_publisher(Float32MultiArray, "igris_c/hand/targets", 10)
        self.create_subscription(String, "igris_c/hand/status", self._status_callback, 10)
        self.create_subscription(
            Float32MultiArray, "igris_c/hand/present_position", self._position_callback, 10
        )
        self._auto_timer: Optional[rclpy.timer.Timer] = None
        self._targets_supplier: Optional[Callable[[], List[float]]] = None
        self._auto_last_time: Optional[float] = None

    def send_command(self, command: str) -> None:
        if not command:
            return
        msg = String()
        msg.data = command
        self._cmd_pub.publish(msg)
        self.get_logger().info(f"Sent command: {command}")

    def send_targets(self, targets: List[float], *, log: bool = True) -> None:
        if len(targets) != len(HAND_MOTOR_IDS):
            self.get_logger().warning("Targets length mismatch; expected 12 values")
            return
        msg = Float32MultiArray()
        msg.data = [float(max(0.0, min(1.0, value))) for value in targets]
        self._targets_pub.publish(msg)
        if log:
            self.get_logger().info("Sent targets")
        else:
            self.get_logger().debug("Sent targets (auto)")

    def start_auto_publish(self, supplier: Callable[[], List[float]], interval: float) -> None:
        self._targets_supplier = supplier
        self._auto_last_time = None
        if self._auto_timer is None:
            self._auto_timer = self.create_timer(interval, self._auto_publish_cb)

    def stop_auto_publish(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.cancel()
            self.destroy_timer(self._auto_timer)
            self._auto_timer = None
        self._targets_supplier = None
        self._auto_last_time = None

    def _auto_publish_cb(self) -> None:
        if self._targets_supplier is None:
            return
        now = time.perf_counter()
        if self._auto_last_time is not None:
            interval_ms = (now - self._auto_last_time) * 1000.0
            self.get_logger().debug(f"auto_publish interval {interval_ms:.2f} ms")
        self._auto_last_time = now
        targets = self._targets_supplier()
        if len(targets) != len(HAND_MOTOR_IDS):
            return
        self.send_targets(targets, log=False)

    def _status_callback(self, msg: String) -> None:
        try:
            self._status_queue.put_nowait(msg.data)
        except queue.Full:
            pass

    def _position_callback(self, msg: Float32MultiArray) -> None:
        try:
            self._position_queue.put_nowait(list(msg.data))
        except queue.Full:
            pass


class HandControlGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("IGRIS C Hand Control")

        self.status_queue: queue.Queue = queue.Queue(maxsize=5)
        self.position_queue: queue.Queue = queue.Queue(maxsize=5)
        self._running = True
        self._poll_job = None
        self._auto_interval = 1.0 / 20.0

        rclpy.init()
        self.node = HandBridgeNode(self.status_queue, self.position_queue)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_job = None

        self.slider_vars: List[tk.DoubleVar] = []
        self.slider_widgets: List[tk.Scale] = []
        self.auto_send = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="No status yet")
        self.position_var = tk.StringVar(value="")
        self._targets_lock = threading.Lock()
        self._targets: List[float] = [0.0] * len(HAND_MOTOR_IDS)

        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_job = self.root.after(50, self._poll_ros_queues)
        self._spin_job = self.root.after(0, self._spin_executor)

    def _build_layout(self) -> None:
        sliders_frame = tk.LabelFrame(self.root, text="Motor Targets")
        sliders_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        for idx, motor_id in enumerate(HAND_MOTOR_IDS):
            var = tk.DoubleVar(value=0.0)
            slider = tk.Scale(
                sliders_frame,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                length=250,
                label=f"ID {motor_id}",
                variable=var,
                command=lambda _val, index=idx: self._on_slider_change(index),
            )
            slider.grid(row=idx, column=0, padx=5, pady=2, sticky="ew")
            self.slider_vars.append(var)
            self.slider_widgets.append(slider)

        control_frame = tk.LabelFrame(self.root, text="Commands")
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        commands = ["init_setup", "move", "move extension", "check", "recover"]
        for idx, command in enumerate(commands):
            btn = tk.Button(
                control_frame,
                text=command,
                width=18,
                command=lambda cmd=command: self.node.send_command(cmd),
            )
            btn.grid(row=idx, column=0, padx=5, pady=2, sticky="ew")

        home_btn = tk.Button(control_frame, text="Home Pose", command=self._set_home_pose)
        home_btn.grid(row=len(commands), column=0, padx=5, pady=(10, 2), sticky="ew")

        targets_btn = tk.Button(control_frame, text="Send Targets", command=self._send_targets)
        targets_btn.grid(row=len(commands) + 1, column=0, padx=5, pady=2, sticky="ew")

        auto_box = tk.Checkbutton(
            control_frame,
            text="Auto send on change",
            variable=self.auto_send,
            command=self._on_auto_toggle,
        )
        auto_box.grid(row=len(commands) + 2, column=0, padx=5, pady=2, sticky="w")

        custom_frame = tk.Frame(control_frame)
        custom_frame.grid(row=len(commands) + 3, column=0, padx=5, pady=(10, 2), sticky="ew")
        tk.Label(custom_frame, text="Custom cmd:").grid(row=0, column=0, padx=2, pady=2)
        self.custom_cmd_var = tk.StringVar()
        custom_entry = tk.Entry(custom_frame, textvariable=self.custom_cmd_var, width=14)
        custom_entry.grid(row=0, column=1, padx=2, pady=2)
        send_custom_btn = tk.Button(custom_frame, text="Send", command=self._send_custom_command)
        send_custom_btn.grid(row=0, column=2, padx=2, pady=2)

        status_frame = tk.LabelFrame(self.root, text="Status")
        status_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        status_label = tk.Label(
            status_frame, textvariable=self.status_var, anchor="w", justify=tk.LEFT
        )
        status_label.pack(fill="x", padx=5, pady=4)

        position_label = tk.Label(
            status_frame, textvariable=self.position_var, anchor="w", justify=tk.LEFT
        )
        position_label.pack(fill="x", padx=5, pady=4)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        sliders_frame.columnconfigure(0, weight=1)

    def _on_slider_change(self, index: int) -> None:
        value = float(self.slider_vars[index].get())
        with self._targets_lock:
            self._targets[index] = value

    def _send_targets(self, *, log: bool = True, update_cache: bool = True) -> None:
        if update_cache:
            with self._targets_lock:
                for idx, var in enumerate(self.slider_vars):
                    self._targets[idx] = float(var.get())
                targets = list(self._targets)
        else:
            with self._targets_lock:
                targets = list(self._targets)
        self.node.send_targets(targets, log=log)

    def _set_home_pose(self) -> None:
        for var in self.slider_vars:
            var.set(0.0)
        self._send_targets(update_cache=True)

    def _send_custom_command(self) -> None:
        cmd = self.custom_cmd_var.get().strip()
        if cmd:
            self.node.send_command(cmd)

    def _on_auto_toggle(self) -> None:
        if self.auto_send.get():
            self._start_auto_send()
        else:
            self._stop_auto_send()

    def _start_auto_send(self) -> None:
        if not self._running:
            return
        self.node.start_auto_publish(self._snapshot_targets, self._auto_interval)

    def _stop_auto_send(self) -> None:
        self.node.stop_auto_publish()

    def _snapshot_targets(self) -> List[float]:
        with self._targets_lock:
            return list(self._targets)

    def _poll_ros_queues(self) -> None:
        updated = False
        while not self.status_queue.empty():
            self.status_var.set(self.status_queue.get())
            updated = True
        while not self.position_queue.empty():
            positions = self.position_queue.get()
            formatted = ", ".join(f"{val:.3f}" for val in positions)
            self.position_var.set(f"Present position: [{formatted}]")
            updated = True
        if updated:
            self.root.update_idletasks()
        if self._running:
            self._poll_job = self.root.after(50, self._poll_ros_queues)

    def _spin_executor(self) -> None:
        if not self._running:
            return
        try:
            self.executor.spin_once(timeout_sec=0.0)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"Executor spin_once failed: {exc}")
        self._spin_job = self.root.after(5, self._spin_executor)

    def on_close(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        if self._spin_job is not None:
            try:
                self.root.after_cancel(self._spin_job)
            except tk.TclError:
                pass
            self._spin_job = None
        self._stop_auto_send()
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.root.destroy()


def main() -> None:
    gui = HandControlGUI()
    gui.root.mainloop()


if __name__ == "__main__":
    main()
