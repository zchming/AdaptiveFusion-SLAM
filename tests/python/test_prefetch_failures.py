"""Failure regression checks for real socket transport; no images, model or GT."""
import json
import socket
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from adaptivefusion.auxiliary_output import CorrectionOutput
from adaptivefusion.prefetch_transport import PrefetchClient


class PrefetchFailureTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = socket.socketpair()
        self.output = CorrectionOutput()
        self.events = []
        self.client = PrefetchClient(self.a, self.output, 5., self.events.append)
        request = dict(schema=1, request_id="match1", scene="probe", source_keyframe=0,
            target_keyframe=1, source_frame=0, target_frame=5, source_timestamp=0., target_timestamp=.2)
        self.client.submit(request, 0., 5)
        self.b.recv(8192)
        self.correction = np.eye(4)
        self.correction[0, 3] = .2
        self.b.sendall((json.dumps(dict(request, status="ok", t_corrected_raw=self.correction.tolist())) + "\n").encode())
        self.client.poll(.01, 5)
        self.prefetch = dict(schema=1, kind="prefetch", request_id="prefetch2", scene="probe",
            source_keyframe=1, source_frame=5, source_timestamp=.2, for_target_keyframe=2,
            visible_frame=5, visible_timestamp=.2, visible_published_monotonic_s=0.)
        self.client.try_prefetch(self.prefetch, .02, 5)
        self.b.recv(8192)

    def tearDown(self):
        self.client.finish(7.)
        self.b.close()

    def assert_original_and_last_correction_preserved(self):
        pose = np.eye(4)
        pose[1, 3] = .3
        emitted = self.output.emit(6, .25, pose)
        np.testing.assert_array_equal(emitted.raw, pose)
        np.testing.assert_array_equal(emitted.corrected, self.correction @ pose)
        self.assertEqual(self.output.request_id, "match1")

    def test_prefetch_failure_keeps_matching_available(self):
        self.b.sendall((json.dumps(dict(self.prefetch, status="failed", reason="feature extraction failed")) + "\n").encode())
        self.client.poll(.03, 5)
        self.assertTrue(self.client.enabled)
        self.assertIsNone(self.client.active)
        self.assertTrue(any(x["status"] == "prefetch_failed" for x in self.events))
        self.assert_original_and_last_correction_preserved()

    def test_disconnect_during_prefetch_keeps_raw_stream_available(self):
        self.b.close()
        self.client.poll(.03, 5)
        self.assertFalse(self.client.enabled)
        self.assert_original_and_last_correction_preserved()

    def test_timeout_during_prefetch_keeps_raw_stream_available(self):
        self.client.poll(5.03, 5)
        self.assertFalse(self.client.enabled)
        self.assertTrue(any(x.get("reason") == "request_timeout" for x in self.events))
        self.assert_original_and_last_correction_preserved()


if __name__ == "__main__":
    unittest.main()
