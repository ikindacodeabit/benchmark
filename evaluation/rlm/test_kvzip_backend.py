# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GPU/dependency preflight checks in kvzip_backend.

Everything here runs without a GPU or torch: query_gpus is monkeypatched, so
the tests exercise the selection/error logic itself — the part that must fail
FAST and CLEARLY instead of OOMing twenty minutes into a run.
"""
import os
import unittest
from unittest import mock

from evaluation.rlm import kvzip_backend


def _gpus(*free_gib):
    return [{"index": i, "free_gib": f, "total_gib": 48.0} for i, f in enumerate(free_gib)]


class PreflightSelectGpuTest(unittest.TestCase):
    def setUp(self):
        # Each test starts unpinned; restore whatever the shell had afterwards.
        self._saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved

    def test_picks_the_freest_gpu_and_pins_it(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(5.0, 30.0, 20.0)):
            chosen = kvzip_backend.preflight_select_gpu(min_free_gib=14.0)
        self.assertEqual(chosen, 1)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_honors_an_explicit_device_when_it_has_room(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(20.0, 30.0)):
            chosen = kvzip_backend.preflight_select_gpu(min_free_gib=14.0, device="cuda:0")
        self.assertEqual(chosen, 0)

    def test_explicit_device_without_room_fails_with_the_per_gpu_picture(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(5.0, 30.0)):
            with self.assertRaisesRegex(RuntimeError, r"GPU 0 has 5\.0 GiB free.*GPU1: 30\.0"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0, device="cuda:0")

    def test_pre_set_cuda_visible_devices_is_respected_not_overridden(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(40.0, 40.0, 20.0, 5.0)):
            chosen = kvzip_backend.preflight_select_gpu(min_free_gib=14.0)
        self.assertEqual(chosen, 2)

    def test_no_gpu_has_room(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(5.0, 8.0)):
            with self.assertRaisesRegex(RuntimeError, "No GPU has"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0)

    def test_no_nvidia_gpu_at_all(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "No NVIDIA GPU visible"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0)

    def test_nonexistent_pinned_gpu(self):
        with mock.patch.object(kvzip_backend, "query_gpus", return_value=_gpus(30.0)):
            with self.assertRaisesRegex(RuntimeError, "GPU 7 does not exist"):
                kvzip_backend.preflight_select_gpu(min_free_gib=14.0, device="cuda:7")


class ImportKvzipTest(unittest.TestCase):
    def test_bad_checkout_path_is_rejected_before_sys_path_pollution(self):
        with self.assertRaisesRegex(RuntimeError, "not a KVzip checkout"):
            kvzip_backend.import_kvzip(kvzip_dir=os.path.dirname(__file__))


if __name__ == "__main__":
    unittest.main()
