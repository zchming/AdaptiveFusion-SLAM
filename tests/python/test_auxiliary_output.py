import json,socket,sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"python"))
from adaptivefusion.auxiliary_output import BoundedAuxiliaryClient,CorrectionOutput

def request(j):
    return dict(schema=1,request_id=f"test:{j}",scene="test",source_keyframe=j-1,target_keyframe=j,
        source_frame=j-1,target_frame=j,source_timestamp=float(j-1),target_timestamp=float(j))

def response(req,correction):
    return dict(req,status="ok",t_corrected_raw=correction.tolist())

class AuxiliaryOutputTests(unittest.TestCase):
    def test_large_world_correction_preserves_raw_pose_and_previous_output(self):
        output=CorrectionOutput();raw=np.eye(4);raw[:3,3]=[1.,2.,3.]
        before=output.emit(0,0.,raw);old=before.corrected.copy()
        correction=np.eye(4);correction[:3,:3]=Rotation.from_euler("z",14.6,degrees=True).as_matrix()
        correction[:3,3]=[.435,0.,0.]
        self.assertTrue(output.accept(response(request(1),correction),request(1),1))
        after=output.emit(1,1.,raw)
        np.testing.assert_array_equal(after.raw,raw)
        np.testing.assert_array_equal(before.corrected,old)
        np.testing.assert_allclose(after.corrected,correction@raw)
        self.assertFalse(np.shares_memory(after.raw,raw))
        self.assertIsNone(output.emit(2,2.,raw,False))
        with self.assertRaises(ValueError):output.emit(1,1.,raw)

    def test_future_stale_and_nonfinite_results_rejected_without_state_change(self):
        output=CorrectionOutput();correction=np.eye(4)
        with self.assertRaises(ValueError):output.accept(response(request(3),correction),request(3),2)
        output.accept(response(request(2),correction),request(2),2)
        with self.assertRaises(ValueError):output.accept(response(request(1),correction),request(1),3)
        correction[0,0]=float("nan")
        with self.assertRaises(ValueError):output.accept(response(request(3),correction),request(3),3)
        self.assertEqual(output.request_id,"test:2")

    def test_socket_queue_budget_and_complete_response(self):
        a,b=socket.socketpair();self.addCleanup(b.close)
        events=[];output=CorrectionOutput();client=BoundedAuxiliaryClient(a,output,5.,events.append)
        self.addCleanup(a.close)
        self.assertTrue(client.submit(request(1),0.,1))
        self.assertTrue(client.submit(request(2),.01,2))
        self.assertFalse(client.submit(request(3),.02,3))
        first=json.loads(b.recv(4096))
        correction=np.eye(4);correction[0,3]=.5
        b.sendall((json.dumps(response(first,correction))+"\n").encode())
        client.poll(.1,3)
        self.assertEqual(output.request_id,"test:1")
        self.assertEqual(json.loads(b.recv(4096))["request_id"],"test:2")
        self.assertEqual(sum(e["status"]=="dropped_queue_full" for e in events),1)
        client.finish(.2)

    def test_timeout_and_disconnect_keep_latest_correction(self):
        for fault in ["timeout","disconnect"]:
            a,b=socket.socketpair();events=[];output=CorrectionOutput()
            correction=np.eye(4);correction[1,3]=.7
            output.accept(response(request(1),correction),request(1),1)
            client=BoundedAuxiliaryClient(a,output,.1,events.append)
            client.submit(request(2),0.,2);b.recv(4096)
            if fault=="disconnect":b.close()
            client.poll(.2 if fault=="timeout" else .01,2)
            self.assertFalse(client.enabled)
            result=output.emit(3,3.,np.eye(4))
            np.testing.assert_allclose(result.corrected,correction)
            np.testing.assert_array_equal(result.raw,np.eye(4))
            b.close();a.close()

    def test_complete_reply_before_eof_is_applied(self):
        a,b=socket.socketpair();events=[];output=CorrectionOutput()
        client=BoundedAuxiliaryClient(a,output,5.,events.append)
        client.submit(request(1),0.,1);b.recv(4096)
        correction=np.eye(4);correction[2,3]=.3
        b.sendall((json.dumps(response(request(1),correction))+"\n").encode());b.close()
        client.poll(.01,1)
        self.assertEqual(output.request_id,"test:1")
        self.assertFalse(client.enabled)
        a.close()

    def test_wrong_request_identity_disables_transport(self):
        a,b=socket.socketpair();self.addCleanup(a.close);self.addCleanup(b.close)
        events=[];output=CorrectionOutput();client=BoundedAuxiliaryClient(a,output,5.,events.append)
        client.submit(request(1),0.,1);b.recv(4096)
        b.sendall((json.dumps(response(request(2),np.eye(4)))+"\n").encode())
        client.poll(.01,2)
        self.assertFalse(client.enabled)
        self.assertIsNone(output.request_id)
if __name__=="__main__":unittest.main()
