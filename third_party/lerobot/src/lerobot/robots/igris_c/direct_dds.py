"""Read-only telemetry plus an explicitly opened LowCmd writer. No igris_core."""

import asyncio
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np


def message_types():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from messages import igris_c_msgs as types

    # Public C++ IDL enums have no explicit default; match their XTypes identifiers.
    for enum in (types.KinematicMode, types.RobotControlState, types.RobotEnvironment):
        enum.__idl_enum_default_value__ = None
    return types


def vector(value, name, positive=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (31,) or not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise ValueError(f"{name}: expected 31 finite {'positive ' if positive else ''}values")
    return array


def lowcmd(types, q, kp, kd, seq, frame):
    q, kp, kd = vector(q, "q"), vector(kp, "kp", True), vector(kd, "kd", True)
    stamp = time.time_ns()
    return types.LowCmd(
        types.make_header(seq, stamp // 10**9, stamp % 10**9, frame),
        [types.KinematicMode.PJS] * 5,
        [types.MotorCmd(i, float(q[i]), 0.0, 0.0, float(kp[i]), float(kd[i])) for i in range(31)],
    )


def require_state(state, now):
    for key, maximum in (("low_at", 0.15), ("robot_at", 2.0), ("web_at", 2.0)):
        age = now - state.get(key, float("-inf"))
        if not 0 <= age <= maximum:
            raise TimeoutError(f"stale {key}: age={age:.3f}s")
    web = state["web"]
    status = web.get("status", {})
    if state["mode"] != 6 or state["environment"] != 0 or web.get("source") != "real":
        raise RuntimeError("REAL WALK_LOW required; prepare with manufacturer console")
    if status.get("initialized") is not True or status.get("torqueOn") is not True:
        raise RuntimeError("robot must already be initialized and torque ON")
    if status.get("estop") is not False:
        raise RuntimeError("E-stop must be clear")
    vector(state["joint_q"], "joint_q")
    vector(web.get("command", {}).get("position"), "controller command position")


class DirectDDS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame = "act-direct-" + uuid.uuid4().hex
        self.writer = None
        self.owner = None
        self.entities = []
        self.state = {}
        self.headers = {}
        self.fault = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.foreign_at = float("-inf")
        self.armed = False

    def connect(self):
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        xml = self.cfg["cyclonedds_xml"]
        os.environ["CYCLONEDDS_URI"] = Path(xml).resolve().as_uri()
        self.types = message_types()
        self.qos = Qos(Policy.Reliability.BestEffort, Policy.Durability.Volatile, Policy.History.KeepLast(5))
        self.participant = DomainParticipant(self.cfg["domain"])
        self.entities.append(self.participant)
        self.readers = {}
        for suffix, cls in (
            ("lowstate", self.types.LowState),
            ("robotstate", self.types.RobotState),
            ("lowcmd", self.types.LowCmd),
        ):
            topic = Topic(self.participant, f"{self.cfg['namespace']}/rt/{suffix}", cls)
            reader = DataReader(self.participant, topic, qos=self.qos)
            self.entities.extend((topic, reader))
            self.readers[suffix] = reader
            if suffix == "lowcmd":
                self.command_topic = topic
        self.thread = threading.Thread(target=self._status_thread, daemon=True)
        self.thread.start()
        return self

    def _status_thread(self):
        try:
            asyncio.run(self._status_loop())
        except Exception as exc:
            self.fault = f"status receiver: {type(exc).__name__}: {exc}"

    async def _status_loop(self):
        from websockets.asyncio.client import connect

        async with connect(self.cfg["status_url"], open_timeout=3, close_timeout=1) as socket:
            while not self.stop.is_set():
                try:
                    raw = await asyncio.wait_for(socket.recv(), 0.2)
                except TimeoutError:
                    continue
                data = json.loads(raw)
                if data.get("type") == "telemetry":
                    with self.lock:
                        self.state.update(web=data, web_at=time.monotonic())

    def poll(self):
        if self.fault:
            raise RuntimeError(self.fault)
        for name, reader in self.readers.items():
            for sample in reader.take(N=100):
                if not sample.sample_info.valid_data:
                    continue
                now = time.monotonic()
                if name == "lowcmd":
                    frame = "".join(sample.header.frame_id).rstrip("\x00")
                    if frame != self.frame:
                        self.foreign_at = now
                        if self.armed:
                            self.fault = "another LowCmd writer; restart required"
                            raise RuntimeError(self.fault)
                    continue
                header = (sample.header.seq, sample.header.sec, sample.header.nanosec)
                if header == self.headers.get(name):
                    continue
                self.headers[name] = header
                with self.lock:
                    if name == "lowstate":
                        self.state.update(
                            joint_q=[j.q for j in sample.joint_state],
                            joint_tau_est=[j.tau_est for j in sample.joint_state],
                            low_at=now,
                        )
                    else:
                        self.state.update(
                            mode=sample.state.value, environment=sample.environment.value, robot_at=now
                        )
        # Endpoint discovery also catches quiet but still alive publishers.
        own = self.writer.instance_handle if self.writer else None
        if any(handle != own for handle in self.readers["lowcmd"].get_matched_publications()):
            self.foreign_at = time.monotonic()
            if self.armed:
                self.fault = "foreign LowCmd endpoint; restart required"
                raise RuntimeError(self.fault)

    def snapshot(self):
        self.poll()
        with self.lock:
            result = self.state.copy()
        require_state(result, time.monotonic())
        return result

    def arm(self, execute):
        self.snapshot()
        if time.monotonic() - self.foreign_at < 3:
            raise RuntimeError("LowCmd must be quiet for three seconds")
        if execute:
            from cyclonedds.pub import DataWriter

            key = hashlib.sha256((str(self.cfg["domain"]) + self.cfg["namespace"]).encode()).hexdigest()[:20]
            self.owner = open(f"/tmp/igris-lowcmd-{key}.lock", "a+")  # noqa: SIM115 -- held until close()
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.writer = DataWriter(self.participant, self.command_topic, qos=self.qos)
        self.armed = True

    def close(self):
        # Detach before waiting on receivers; no mode, torque, HOLD or hand services.
        if self.writer is not None:
            self.writer.__del__()
            self.writer = None
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=4)
        for entity in reversed(self.entities):
            entity.__del__()
        self.entities.clear()
        if self.owner:
            self.owner.close()
            self.owner = None
