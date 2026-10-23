import unittest
import numpy as np
from adaptivefusion.matching_verification import checked_mask, filtered_rows, equal_count_rows


class MatchingVerificationTests(unittest.TestCase):
    def test_full_2d_mask_maps_onto_depth_rows_without_renumbering(self):
        matches = np.array([[2, 8], [5, 11], [9, 3]])
        rows = [dict(source_feature=2, target_feature=8), dict(source_feature=9, target_feature=3)]
        selected = filtered_rows(matches, np.array([False, True, True]), rows)
        self.assertEqual(selected, [rows[1]])
        with self.assertRaisesRegex(ValueError, 'absent'):
            filtered_rows(matches, [1, 1, 1], [dict(source_feature=7,target_feature=8)])

    def test_equal_count_controls_preserve_ids_order_and_cardinality(self):
        rows = [dict(source_feature=2*i+1) for i in range(40)]
        a = equal_count_rows(rows, 13, 17); b = equal_count_rows(rows, 13, 23)
        self.assertEqual(a, equal_count_rows(rows, 13, 17))
        self.assertEqual(len(a), 13); self.assertNotEqual(a, b)
        self.assertEqual([r['source_feature'] for r in a], sorted(r['source_feature'] for r in a))
        self.assertEqual(equal_count_rows(rows, 0, 17), [])

    def test_failed_or_misaligned_model_is_not_accepted(self):
        with self.assertRaises(ValueError): checked_mask(None, None, 8)
        with self.assertRaises(ValueError): checked_mask(np.eye(3), [1, 1], 8)
        model, mask = checked_mask(np.eye(3), [0,1], 2)
        self.assertEqual(mask.tolist(), [False, True])


if __name__ == '__main__': unittest.main()
