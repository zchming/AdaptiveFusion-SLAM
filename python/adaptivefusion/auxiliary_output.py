"""Asynchronous auxiliary transport and a correction-only pose output layer."""
from dataclasses import dataclass
import json
import select
import socket
import numpy as np
from .geometry import check_pose

@dataclass(frozen=True)
class FrameOutput:
    frame_id: int
    timestamp: float
    raw: np.ndarray
    corrected: np.ndarray
    correction_request: str | None

class CorrectionOutput:
    """T_corrected_camera = T_corrected_raw @ T_raw_camera; never edits input."""
    def __init__(self):
        self._correction = np.eye(4)
        self.request_id = None
        self.target_frame = -1
        self.last_frame = -1

    def accept(self, result, request, visible_frame):
        if not isinstance(result,dict) or result.get("schema") != 1 or result.get("request_id") != request["request_id"]:
            raise ValueError("response schema/request identity mismatch")
        for field in ["scene","source_keyframe","target_keyframe","source_frame","target_frame",
                      "source_timestamp","target_timestamp"]:
            if result.get(field) != request[field]:
                raise ValueError("response ownership mismatch: "+field)
        if request["target_frame"] > visible_frame or request["target_frame"] <= self.target_frame:
            raise ValueError("future or stale correction")
        if result.get("status") != "ok":
            return False
        correction = check_pose(np.asarray(result["t_corrected_raw"],float)).copy()
        self._correction = correction
        self.request_id = request["request_id"]
        self.target_frame = request["target_frame"]
        return True

    def emit(self, frame_id, timestamp, raw_pose, valid=True):
        if frame_id <= self.last_frame:
            raise ValueError("output frame order/revision")
        self.last_frame = frame_id
        if not valid:
            return None
        raw = check_pose(raw_pose).copy()
        corrected = check_pose(self._correction@raw).copy()
        raw.setflags(write=False)
        corrected.setflags(write=False)
        return FrameOutput(frame_id,timestamp,raw,corrected,self.request_id)

class BoundedAuxiliaryClient:
    """One in-flight message plus one local waiting request; poll never waits."""
    def __init__(self, connection, output, timeout_s, event):
        self.socket = connection
        self.socket.setblocking(False)
        self.output = output
        self.timeout_s = timeout_s
        self.event = event
        self.active = None
        self.pending = None
        self.outgoing = b""
        self.incoming = b""
        self.enabled = True
        self.visible_frame = -1
        self.maximum_waiting = 0

    def _start(self, request, now):
        self.active = dict(request=request,dispatch_s=now)
        self.outgoing = (json.dumps(request,allow_nan=False)+"\n").encode()
        self.event(dict(status="dispatched",request_id=request["request_id"],at_s=now))

    def _flush(self):
        if self.outgoing:
            try:
                sent = self.socket.send(self.outgoing)
                self.outgoing = self.outgoing[sent:]
            except BlockingIOError:
                pass

    def disable(self, reason, now):
        if not self.enabled:
            return
        for entry in [self.active,dict(request=self.pending) if self.pending else None]:
            if entry:
                self.event(dict(status="service_unavailable",request_id=entry["request"]["request_id"],
                                reason=reason,at_s=now))
        self.active = self.pending = None
        self.enabled = False
        self.event(dict(status="service_disabled",reason=reason,at_s=now))
        self.socket.close()

    def submit(self, request, now, visible_frame):
        self.visible_frame = max(self.visible_frame,visible_frame)
        if request["target_frame"] > self.visible_frame:
            raise ValueError("future request")
        if not self.enabled:
            self.event(dict(status="not_submitted_service_unavailable",request_id=request["request_id"],at_s=now))
            return False
        if self.active is None:
            self._start(request,now)
        elif self.pending is None:
            self.pending = request
            self.maximum_waiting = 1
            self.event(dict(status="queued",request_id=request["request_id"],at_s=now))
        else:
            self.event(dict(status="dropped_queue_full",request_id=request["request_id"],at_s=now))
            return False
        try:
            self._flush()
        except OSError as error:
            self.disable(str(error),now)
            return False
        return True

    def poll(self, now, visible_frame):
        self.visible_frame = max(self.visible_frame,visible_frame)
        if not self.enabled:
            return
        if self.active and now-self.active["dispatch_s"] > self.timeout_s:
            self.disable("request_timeout",now)
            return
        try:
            self._flush()
            eof = False
            while True:
                try:
                    data = self.socket.recv(65536)
                except BlockingIOError:
                    break
                if not data:
                    eof = True
                    break
                self.incoming += data
                if len(self.incoming) > 1024*1024:
                    raise ValueError("response budget exceeded")
            while b"\n" in self.incoming:
                line,self.incoming = self.incoming.split(b"\n",1)
                result = json.loads(line)
                if self.active is None:
                    raise ValueError("unsolicited/duplicate result")
                request = self.active["request"]
                applied = self.output.accept(result,request,self.visible_frame)
                self.event(dict(status="delivered" if applied else "auxiliary_failed",
                    request_id=request["request_id"],at_s=now,dispatch_s=self.active["dispatch_s"],
                    result=result))
                self.active = None
                if self.pending:
                    request = self.pending
                    self.pending = None
                    self._start(request,now)
                    self._flush()
            if eof:
                self.disable("worker_disconnected",now)
        except (OSError,ValueError,KeyError,TypeError) as error:
            self.disable("invalid_or_failed_transport: "+str(error),now)

    def finish(self, now):
        for entry in [self.active,dict(request=self.pending) if self.pending else None]:
            if entry:
                self.event(dict(status="unfinished_at_stream_end",
                    request_id=entry["request"]["request_id"],at_s=now))
        self.active = self.pending = None
        self.enabled = False
        self.socket.close()
