"""Idle-only feature messages sharing the existing bounded pose transport."""
import json

from .auxiliary_output import BoundedAuxiliaryClient


class PrefetchClient(BoundedAuxiliaryClient):
    def try_prefetch(self, request, now, visible_frame):
        self.visible_frame = max(self.visible_frame, visible_frame)
        if request["source_frame"] > self.visible_frame:
            raise ValueError("future source prefetch")
        if not self.enabled:
            self.event(dict(status="prefetch_skipped_unavailable", request_id=request["request_id"], at_s=now))
            return False
        if self.active is not None or self.pending is not None:
            self.event(dict(status="prefetch_skipped_busy", request_id=request["request_id"], at_s=now))
            return False
        self._start(request, now)
        try:
            self._flush()
        except OSError as error:
            self.disable(str(error), now)
            return False
        return True

    def poll(self, now, visible_frame):
        self.visible_frame = max(self.visible_frame, visible_frame)
        if not self.enabled:
            return
        if self.active and now - self.active["dispatch_s"] > self.timeout_s:
            self.disable("request_timeout", now)
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
                if len(self.incoming) > 1024 * 1024:
                    raise ValueError("response budget exceeded")
            while b"\n" in self.incoming:
                line, self.incoming = self.incoming.split(b"\n", 1)
                result = json.loads(line)
                if self.active is None:
                    raise ValueError("unsolicited/duplicate result")
                request = self.active["request"]
                if request.get("kind") == "prefetch":
                    for key in ["schema", "kind", "request_id", "scene", "source_keyframe",
                                "source_frame", "source_timestamp", "for_target_keyframe",
                                "visible_frame", "visible_timestamp", "visible_published_monotonic_s"]:
                        if result.get(key) != request[key]:
                            raise ValueError("prefetch response ownership mismatch: " + key)
                    if request["source_frame"] > self.visible_frame:
                        raise ValueError("future prefetch response")
                    status = "prefetch_complete" if result.get("status") == "ok" else "prefetch_failed"
                else:
                    applied = self.output.accept(result, request, self.visible_frame)
                    status = "delivered" if applied else "auxiliary_failed"
                self.event(dict(status=status, request_id=request["request_id"], at_s=now,
                                dispatch_s=self.active["dispatch_s"], result=result))
                self.active = None
                if self.pending:
                    request = self.pending
                    self.pending = None
                    self._start(request, now)
                    self._flush()
            if eof:
                self.disable("worker_disconnected", now)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.disable("invalid_or_failed_transport: " + str(error), now)

    def finish(self, now):
        if self.active and self.active["request"].get("kind") == "prefetch":
            self.event(dict(status="unfinished_prefetch_at_stream_end",
                            request_id=self.active["request"]["request_id"], at_s=now))
            self.active = None
        super().finish(now)
