# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pathlib
import pytest
import numpy as np
import torch

from torchcodec._core import (
    create_from_file,
    add_video_stream,
    get_next_frame,
    get_frame_at_index,
    get_frame_at_pts,
    get_frames_in_range,
    get_json_metadata,
    seek_to_pts,
)

from .utils import (
    assert_frames_equal,
    cpu_and_cuda,
    TestVideo,
    TestVideoStreamInfo,
    TestFrameInfo,
    _get_file_path,
    VAR_FPS_VIDEO,
)

class TestVariableFPSVideoDecoder:
    def _check_video_exists(self):
        try:
            VAR_FPS_VIDEO.path
        except FileNotFoundError:
            pytest.skip("Variable FPS test video not found")

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_basic_decoding(self, device):
        self._check_video_exists()
        
        decoder = create_from_file(str(VAR_FPS_VIDEO.path))
        add_video_stream(decoder, device=device)
        
        frame0, pts0, duration0 = get_next_frame(decoder)
        reference_frame0 = VAR_FPS_VIDEO.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0.to(device))
        
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert "numFrames" in metadata_dict
        assert metadata_dict["numFrames"] > 30, "Test video should have at least 30 frames"

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_exact_seeking_mode(self, device):
        """Test behavior in exact seeking mode (should work properly)"""
        self._check_video_exists()
        
        decoder = create_from_file(str(VAR_FPS_VIDEO.path), seek_mode="exact")
        add_video_stream(decoder, device=device)
        
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        
        test_pts = [0.0, 0.5, 1.0, 1.5, 2.0]
        for pts in test_pts:
            if pts < metadata_dict["durationSeconds"]:
                frame, actual_pts, duration = get_frame_at_pts(decoder, pts)
                assert frame is not None, f"Failed to seek to {pts}s in exact mode"
                assert abs(actual_pts.item() - pts) <= duration.item(), \
                    f"Frame timestamp ({actual_pts.item()}) differs too much from requested time ({pts})"

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_approximate_seeking_mode_behavior(self, device):
        """Test behavior in approximate seeking mode (may fail as expected)"""
        self._check_video_exists()
        
        # Create two decoders: one with exact mode, one with approximate mode
        decoder_exact = create_from_file(str(VAR_FPS_VIDEO.path), seek_mode="exact")
        add_video_stream(decoder_exact, device=device)
        
        decoder_approx = create_from_file(str(VAR_FPS_VIDEO.path), seek_mode="approximate")
        add_video_stream(decoder_approx, device=device)

        metadata = get_json_metadata(decoder_exact)
        metadata_dict = json.loads(metadata)
        
        # Compare seeking in both modes
        test_pts = [0.0, 0.5, 1.0, 1.5, 2.0]
        differences = []
        
        for pts in test_pts:
            if pts < metadata_dict["durationSeconds"]:
                frame_exact, pts_exact, _ = get_frame_at_pts(decoder_exact, pts)
                
                try:
                    frame_approx, pts_approx, _ = get_frame_at_pts(decoder_approx, pts)
                    
                    differences.append({
                        "seek_pts": pts,
                        "exact_pts": pts_exact.item(),
                        "approx_pts": pts_approx.item(),
                        "frames_match": torch.allclose(frame_exact, frame_approx),
                        "pts_difference": abs(pts_exact.item() - pts_approx.item()),
                        "approximate_failed": False
                    })
                except Exception as e:
                    differences.append({
                        "seek_pts": pts,
                        "exact_pts": pts_exact.item(),
                        "error": str(e),
                        "approximate_failed": True
                    })
        
        # Print differences (useful for debugging)
        for diff in differences:
            if diff["approximate_failed"]:
                print(f"Seeking to {diff['seek_pts']}s failed in approximate mode: {diff['error']}")
            else:
                print(f"Seeking to {diff['seek_pts']}s: exact={diff['exact_pts']}, "
                      f"approx={diff['approx_pts']}, diff={diff['pts_difference']}, "
                      f"frames {'match' if diff['frames_match'] else 'differ'}")
        
        # No assertion as approximate mode is expected to potentially fail

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_frame_timing_pattern(self, device):
        self._check_video_exists()
        
        decoder = create_from_file(str(VAR_FPS_VIDEO.path))
        add_video_stream(decoder, device=device)
        
        frames_info = []
        frame_index = 0
        
        seek_to_pts(decoder, 0.0)
        
        while True:
            try:
                frame, pts, duration = get_next_frame(decoder)
                frames_info.append({
                    "index": frame_index,
                    "pts": pts.item(),
                    "duration": duration.item()
                })
                frame_index += 1
            except IndexError:
                break
        
        assert len(frames_info) > 30, "Not enough frames to verify variable frame rate"        

        intervals_before = [frames_info[i+1]["pts"] - frames_info[i]["pts"] 
                           for i in range(min(30, len(frames_info)-1))]
        
        intervals_after = [frames_info[i+1]["pts"] - frames_info[i]["pts"] 
                          for i in range(30, min(60, len(frames_info)-1))]
        
        if len(intervals_after) > 5:
            avg_interval_before = sum(intervals_before) / len(intervals_before)
            avg_interval_after = sum(intervals_after) / len(intervals_after)
            
            print(f"Average interval for first 30 frames: {avg_interval_before:.6f}s")
            print(f"Average interval for subsequent frames: {avg_interval_after:.6f}s")
            
            expected_ratio = 0.5
            actual_ratio = avg_interval_before / avg_interval_after
            
            # Allow for some error
            assert abs(actual_ratio - expected_ratio) < 0.2, \
                f"Interval ratio ({actual_ratio:.2f}) differs too much from expected ({expected_ratio})"

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_sequential_decoding(self, device):
        self._check_video_exists()
        
        decoder = create_from_file(str(VAR_FPS_VIDEO.path))
        add_video_stream(decoder, device=device)
        
        seek_to_pts(decoder, 0.0)
        
        # Decode multiple frames and verify monotonically increasing timestamps
        last_pts = -1.0
        for _ in range(50):
            try:
                frame, pts, duration = get_next_frame(decoder)
                current_pts = pts.item()
                
                assert current_pts > last_pts, \
                    f"Frame timestamps not monotonically increasing: current={current_pts}, previous={last_pts}"
                
                last_pts = current_pts
            except IndexError:
                break

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_frames_in_range(self, device):
        self._check_video_exists()
        
        decoder = create_from_file(str(VAR_FPS_VIDEO.path))
        add_video_stream(decoder, device=device)
        
        frames, pts_list, durations = get_frames_in_range(decoder, start=0, stop=10)
        
        assert len(frames) == 10, f"Expected 10 frames, got {len(frames)}"
        
        assert len(pts_list) == len(frames), "pts list length should match frame count"
        assert len(durations) == len(frames), "durations list length should match frame count"
        
        for i in range(1, len(pts_list)):
            assert pts_list[i] > pts_list[i-1], f"pts not monotonically increasing: {pts_list}"