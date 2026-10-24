import unittest
import numpy as np
from adaptivefusion.depth_time_alignment import estimated_pose, sensor_motion, register_source_depth
from adaptivefusion.trajectory import FrameRecord


class DepthTimeAlignmentTests(unittest.TestCase):
    def test_identity_preserves_depth_and_holes_exactly(self):
        depth = np.array([[0, 5000, 10000], [65535, 1, 0]], np.uint16)
        warped, _ = register_source_depth(depth, [517.3, 516.5, 1, 1], np.eye(4))
        np.testing.assert_array_equal(warped, depth)

    def test_forward_projection_and_z_buffer(self):
        depth = np.array([[5000, 10000], [0, 0]], np.uint16)
        T = np.eye(4); T[2, 3] = 10.
        warped, stats = register_source_depth(depth, [1, 1, 0, 0], T)
        self.assertEqual(warped[0, 0], 55000)  # nearer point wins, in 1/5000m units
        self.assertEqual(np.count_nonzero(warped), 1)
        self.assertEqual(stats['collision_count'], 1)

    def test_motion_direction_uses_depth_to_rgb(self):
        a, b = np.eye(4), np.eye(4); b[0, 3] = .03
        frames = [FrameRecord(0, 1., True, -1, 0, a), FrameRecord(1, 1.03, True, 0, 0, b)]
        motion, _ = sensor_motion(frames, 1.03, 1.015)
        self.assertAlmostEqual(motion[0, 3], -.015)

    def test_rotation_interpolates_and_lost_frames_are_not_filled(self):
        from scipy.spatial.transform import Rotation
        a, b = np.eye(4), np.eye(4); b[:3, :3] = Rotation.from_euler('z', 20, degrees=True).as_matrix()
        frames = [FrameRecord(0, 1., True, -1, 0, a), FrameRecord(1, 1.04, True, 0, 0, b)]
        p, _ = estimated_pose(frames, 1.02)
        self.assertAlmostEqual(Rotation.from_matrix(p[:3,:3]).as_euler('xyz', degrees=True)[2], 10.)
        frames[0].valid = False
        with self.assertRaisesRegex(ValueError, 'lost'): estimated_pose(frames, 1.02)

    def test_extrapolation_rejects_and_optional_gap_limit_is_explicit(self):
        frames = [FrameRecord(0, 1., True, -1, 0, np.eye(4)), FrameRecord(1, 1.1, True, 0, 0, np.eye(4))]
        with self.assertRaisesRegex(ValueError, 'bracket'): estimated_pose(frames, .99)
        p, ids = estimated_pose(frames, 1.02)
        self.assertEqual(ids, [0, 1])
        with self.assertRaisesRegex(ValueError, 'configured gap'): estimated_pose(frames, 1.02, maximum_gap_s=.05)


if __name__ == '__main__': unittest.main()
