"""Durable, operator-requested uploads of finalized web inference recordings."""

import copy
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def push_dataset(entry):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(entry["dataset_root"])
    evidence = root / "rollout"
    evidence.mkdir(exist_ok=True)
    for name in ("config.json", "result.json", "recording_result.json", "commands.jsonl"):
        source = Path(entry["output_dir"]) / name
        if source.is_file():
            shutil.copy2(source, evidence / name)
    dataset = LeRobotDataset(entry["repo_id"], root=root, video_backend="pyav")
    if dataset.num_frames != entry["frames"] or dataset.num_episodes != entry["saved_episodes"]:
        raise ValueError("Finalized dataset counts differ from the recording result; upload refused")
    dataset.push_to_hub(private=True, tags=["igris", "policy-inference", "model-annotated"])


class RecordingStore:
    """One web process owns this index. Network IO never holds the store/web lock."""

    def __init__(self, root, owner, *, uploader=push_dataset):
        if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", owner):
            raise ValueError("Set --rollout-hub-owner to your Hugging Face user or organization")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.uploader = uploader
        self.lock = threading.RLock()
        self.entries = {}
        self.active = None
        self.thread = None
        self.closed = False
        for path in sorted(self.root.glob("*.json")):
            entry = json.loads(path.read_text())
            if entry["repo_id"].split("/")[0] != owner:
                continue
            self.entries[entry["id"]] = entry
            if entry["status"] == "active":
                self._finalize(entry, {})
            elif entry["status"] == "uploading":
                entry.update(status="error", error="업로드 중 웹이 종료되었습니다. 버튼을 눌러 재시도하세요.")
            entry["requested"] = False
            self._save(entry)

    def _save(self, entry):
        path = self.root / (entry["id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    def begin(self, output, policy_id, policy_name):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("A recording is still being finalized")
            identity = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
            entry = dict(id=identity, repo_id=f"{self.owner}/igris-inference-{identity}",
                         output_dir=str(Path(output).resolve()), policy_id=policy_id, policy_name=policy_name,
                         status="active", frames=0, saved_episodes=0, error=None, requested=False, private=True)
            self.entries[identity] = entry
            self.active = identity
            self._save(entry)
            return copy.deepcopy(entry)

    def _finalize(self, entry, result):
        path = Path(entry["output_dir"]) / "recording_result.json"
        recording = json.loads(path.read_text()) if path.exists() else result.get("recording")
        entry["run_error"] = result.get("error")
        if recording is None:
            entry.update(status="invalid", error="녹화 마감 결과가 없습니다. 로컬 실행 로그를 확인하세요.")
        elif recording.get("error") or recording.get("encoder_dropped_frames"):
            entry.update(status="invalid", error=recording.get("error") or "영상 프레임 누락")
        elif not recording.get("frames") or not recording.get("saved_episodes"):
            entry.update(status="empty", error=None)
        else:
            entry.update(status="ready", error=None, frames=recording["frames"],
                         saved_episodes=recording["saved_episodes"], dataset_root=recording["dataset_root"])

    def finish(self, result):
        with self.lock:
            if self.active is not None:
                entry = self.entries[self.active]
                self._finalize(entry, result)
                self._save(entry)
                self.active = None
            self._start_uploads()

    def request_upload(self):
        with self.lock:
            for entry in self.entries.values():
                if entry["status"] in ("active", "ready", "error"):
                    entry["requested"] = True
                    self._save(entry)
            self._start_uploads()

    def _start_uploads(self):
        if self.closed or self.active is not None or self.thread is not None:
            return
        if not any(e["requested"] and e["status"] in ("ready", "error") for e in self.entries.values()):
            return
        self.thread = threading.Thread(target=self._upload, daemon=True, name="act-hub-upload")
        self.thread.start()

    def _upload(self):
        while True:
            with self.lock:
                entry = next((e for e in self.entries.values()
                              if e["requested"] and e["status"] in ("ready", "error")), None)
                if self.closed or self.active is not None or entry is None:
                    self.thread = None
                    return
                entry.update(status="uploading", requested=False, error=None)
                self._save(entry)
                payload = copy.deepcopy(entry)
            try:
                self.uploader(payload)
            except Exception as exc:
                update = dict(status="error", error=f"{type(exc).__name__}: {exc}")
            else:
                update = dict(status="uploaded", error=None,
                              url=f"https://huggingface.co/datasets/{entry['repo_id']}")
            with self.lock:
                entry.update(update)
                self._save(entry)

    def snapshot(self):
        with self.lock:
            entries = copy.deepcopy(list(self.entries.values()))
            return dict(entries=entries, active=self.active, private=True,
                        finishing=bool(self.active and self.entries[self.active]["requested"]),
                        can_upload=any(e["status"] in ("active", "ready", "error") for e in entries))

    def wait(self, timeout):
        with self.lock:
            thread = self.thread
        if thread is not None:
            thread.join(timeout)

    def close(self):
        with self.lock:
            self.closed = True
        # Interrupted uploads remain in the durable index for an explicit retry.
        self.wait(1)


def recording_web_class(base):
    class RecordingWeb(base):
        def __init__(self, cfg, output, **kwargs):
            self.recordings = RecordingStore(cfg["recording_index"], cfg["rollout_hub_owner"])
            self.finishing_recording = False
            super().__init__(cfg, output, **kwargs)

        def prepare_recording_run(self, cfg, output, policy_id):
            with self.lock:
                entry = self.recordings.begin(output, policy_id, self.policies[policy_id]["name"])
                cfg["recording_repo_id"] = entry["repo_id"]
                # A click immediately after Start can precede recorder construction.
                if self.finishing_recording:
                    self.recordings.request_upload()

        def actions(self):
            actions = super().actions()
            if self.shutdown_requested.is_set():
                return actions
            if self.finishing_recording:
                return [a for a in actions if a in ("stop", "shutdown")]
            if self.started.is_set() or self.recordings.snapshot()["can_upload"]:
                actions = [*actions, "finish_upload"]
            return actions

        def _can_select(self):
            return not self.finishing_recording and super()._can_select()

        def command(self, action):
            if action != "finish_upload":
                return super().command(action)
            with self.lock:
                if self.shutdown_requested.is_set():
                    raise RuntimeError("웹이 종료 중입니다.")
                if self.started.is_set():
                    self.finishing_recording = True
                    self.stopped.set()
                    self.pending = "finish_upload"
                if not self.started.is_set() or self.recordings.snapshot()["active"] is not None:
                    self.recordings.request_upload()

        def finish_run(self, result):
            # Recorder.close() and result.json precede this hook in execute_run.
            with self.lock:
                self.recordings.finish(result)
                super().finish_run(result)
                self.finishing_recording = False

        def snapshot(self):
            result = super().snapshot()
            result["recording"] = self.recordings.snapshot()
            result["recording"]["finishing"] = self.finishing_recording
            return result

        def close(self):
            try:
                super().close()
            finally:
                self.recordings.close()

    return RecordingWeb


def recording_page(page):
    for marker in ('<button data-action="stop"', '<div class="layout">', 'function draw(s){', '</script>'):
        if page.count(marker) != 1:
            raise ValueError(f"Recording UI requires one markup marker: {marker}")
    button = '<button data-action="finish_upload" disabled>종료 및 업로드</button>'
    panel = ('<section class="card"><h2>추론 기록 · LeRobot</h2>'
             '<p class="sub" id="recording-status">추론 시 자동 기록 · 비공개 Hub 업로드</p>'
             '<p class="sub">일시정지 중에는 기록을 멈춥니다. 종료 및 업로드를 누르면 제어와 기록을 종료하고 '
             '저장된 미업로드 기록을 업로드합니다. 실패 시 같은 버튼으로 재시도할 수 있습니다.</p>'
             '<div id="recording-runs" class="sub"></div></section>')
    script = """
function drawRecording(s){
 const rec=s.recording;if(!rec)return;
 const current=(s.policies||[]).find(p=>p.id===s.policy_id);
 put('recording-status',rec.finishing?'제어 종료 · 기록 마감 중':rec.active?
   (s.phase==='policy'?'● 기록 중 · ':'기록 대기 · ')+(current?.name||s.policy_id):'추론 시 자동 기록 · 비공개 Hub 업로드');
 const states={active:'기록 세션',ready:'저장 완료 · 업로드 대기',uploading:'Hub 업로드 중',uploaded:'업로드 완료',
   error:'업로드 실패 · 재시도 가능',empty:'추론 프레임 없음',invalid:'기록 오류 · 로컬 보존'};
 const container=document.getElementById('recording-runs');container.replaceChildren();
 for(const e of [...rec.entries].reverse()){
   const row=document.createElement('p');row.textContent='시작 모델 '+e.policy_name+' · '+(states[e.status]||e.status)+
     (e.frames?' · '+e.frames+' frames':'')+(e.error?' · '+e.error:'');
   if(e.url){const a=document.createElement('a');a.href=e.url;a.target='_blank';a.rel='noopener noreferrer';
     a.textContent=' '+e.repo_id;row.append(a);}container.append(row);
 }
}
"""
    page = page.replace('<button data-action="stop"', button + '<button data-action="stop"', 1)
    page = page.replace('<div class="layout">', panel + '<div class="layout">', 1)
    page = page.replace('function draw(s){', 'function draw(s){drawRecording(s);', 1)
    return page.replace('</script>', script + '</script>', 1)
