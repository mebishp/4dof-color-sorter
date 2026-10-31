#!/usr/bin/env python3
"""System Test: End-to-End Sorting Result.

Reads sorting_metrics.csv from the last ./run.sh sort / demo and verifies:
  1. CSV has the expected columns.
  2. One row per block in the current world (groundtruth.json).
  3. Every block was SORTED_SUCCESS with verified=True: the controller only
     reports success after the bin pose sensor saw the block at rest inside
     its own color's bin.
"""

import csv
import json
import os
import unittest

CSV_PATH = '/home/hyprnon/ros_ws/sorting_metrics.csv'
GT_PATH = '/home/hyprnon/ros_ws/worlds/groundtruth.json'
FIELDS = ['color', 'status', 'material', 'pick_pos', 'drop_pos', 'cycle_time_sec',
          'attempts', 'final_pos', 'verified']


class TestSortingE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CSV_PATH):
            raise unittest.SkipTest(f"{CSV_PATH} does not exist yet. Run a sort first.")
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            cls.header = reader.fieldnames or []
            cls.rows = list(reader)

    def test_metrics_csv_structure(self):
        for field in FIELDS:
            self.assertIn(field, self.header, f"Field '{field}' missing from CSV header")

    def test_one_row_per_block(self):
        if not os.path.exists(GT_PATH):
            self.skipTest("groundtruth.json missing")
        with open(GT_PATH) as f:
            gt = json.load(f)
        if os.path.getmtime(GT_PATH) > os.path.getmtime(CSV_PATH):
            self.skipTest("World was regenerated after the last sort; CSV is from an older world")
        self.assertEqual(sorted(r['color'] for r in self.rows), sorted(gt))

    def test_all_blocks_verified_in_bins(self):
        self.assertGreater(len(self.rows), 0, "No entries in sorting_metrics.csv")
        for row in self.rows:
            self.assertEqual(row['status'], 'SORTED_SUCCESS',
                             f"{row['color']}: status={row['status']} final_pos={row.get('final_pos')}")
            self.assertEqual(row.get('verified'), 'True', f"{row['color']}: placement not verified")


if __name__ == '__main__':
    unittest.main()
