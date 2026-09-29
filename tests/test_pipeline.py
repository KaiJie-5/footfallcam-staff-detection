"""Regression tests: no GPU, weights, network, or production video required."""
import csv
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from src.badge import StaffTagDetector
from src.detector import CorridorZoneFilter, restore_boxes
from src.pipeline_runner import export_results, parse_arguments
from src.temporal import ConsensusConfig, classify_tracks, compress_frame_ranges
from src.video_analyzer import analyze_video_stream, detect_motion_intervals
from src.video_io import FrameQuality, validate_video_metadata


def make_frames(count, scores=None, invalid=(), missing=()):
    scores=scores or {}
    result=[]
    for i in range(count):
        d=dict(track_id=1,bbox=[10,10,110,210],center=[60.,110.],conf=.8,
               badge=None if i not in scores else dict(score=scores[i],bbox=[50,80,70,95],contrast=80.,angle=0,template='test'))
        result.append(dict(frame_id=i,quality=dict(valid=i not in invalid,reason='ok' if i not in invalid else 'blackout'),
                           detections=[] if i in missing else [d]))
    return result


@pytest.mark.parametrize('angle,rotated',[
    (0,[10,20,30,50]),(90,[20,70,50,90]),
    (180,[70,30,90,60]),(270,[30,10,60,30]),
])
def test_inverse_rotation_in_non_square_image(angle,rotated):
    np.testing.assert_allclose(restore_boxes([rotated],angle,100,80),[[10,20,30,50]])


def test_five_scattered_hits_do_not_label_a_long_track():
    frames=make_frames(501,{0:.96,100:.96,200:.96,300:.96,400:.96})
    audits=classify_tracks(frames,25)
    assert sum(a['staff_frames'] for a in audits)==0


def test_stationary_staff_is_allowed_and_propagation_is_bounded():
    frames=make_frames(200,{50:.99,55:.86,60:.99})
    classify_tracks(frames,25)
    assert frames[50]['detections'][0]['status']=='staff'
    assert frames[135]['detections'][0]['status']=='staff'
    assert frames[136]['detections'][0]['status']=='unknown'


def test_blackout_breaks_identity_evidence():
    frames=make_frames(40,{0:.99,5:.86,10:.99},invalid={15})
    classify_tracks(frames,25)
    assert frames[10]['detections'][0]['status']=='staff'
    assert frames[15]['detections'][0]['status']=='unknown'
    assert frames[16]['detections'][0]['status']=='unknown'


def test_large_track_gap_does_not_transfer_staff_label():
    frames=make_frames(50,{0:.99,5:.86,10:.99},missing=set(range(15,30)))
    classify_tracks(frames,25)
    assert frames[30]['detections'][0]['status']=='unknown'
    assert frames[0]['detections'][0]['segment_id']!=frames[30]['detections'][0]['segment_id']


def test_no_fabricated_coordinates_in_missing_frames(tmp_path):
    frames=make_frames(20,{0:.99,5:.86,10:.99},missing={6,7})
    classify_tracks(frames,25)
    task1,rows=export_results(frames,dict(width=200,height=240,fps=25),tmp_path,CorridorZoneFilter())
    assert 6 not in task1['frames'] and 7 not in task1['frames']
    assert {r['frame_id'] for r in rows}==set(task1['frames'])
    assert all(r['position_source']=='observed_box_center' for r in rows)


def test_roi_is_resolution_independent():
    roi=CorridorZoneFilter([[.25,.25],[.75,.25],[.75,.75],[.25,.75]])
    assert roi.is_inside(50,50,100,100)
    assert roi.is_inside(500,500,1000,1000)
    assert not roi.is_inside(10,50,100,100)
    with pytest.raises(ValueError):
        CorridorZoneFilter([[10,10],[20,20],[10,20]])


def test_empty_output_still_has_csv_header(tmp_path):
    frames=make_frames(2)
    classify_tracks(frames,25)
    task1,rows=export_results(frames,dict(width=200,height=240,fps=25),tmp_path,CorridorZoneFilter())
    assert task1['frames']==[] and rows==[]
    with (tmp_path/'staff_trajectories.csv').open() as f:
        reader=csv.DictReader(f)
        assert 'frame_id' in reader.fieldnames
        assert list(reader)==[]


def test_blackout_recovery_does_not_use_hardcoded_frame_indices():
    quality=FrameQuality()
    for _ in range(8):
        assert quality.inspect(np.full((20,20,3),100,np.uint8))['valid']
    for level in (5,30,45):
        assert not quality.inspect(np.full((20,20,3),level,np.uint8))['valid']
    assert quality.inspect(np.full((20,20,3),96,np.uint8))['valid']


def test_sensor_glitch_is_not_a_motion_interval():
    df=pd.DataFrame(dict(frame_id=[0,5,10,15,20],valid=[True,True,False,True,True],motion_score=[1.,1.,100.,10.,11.]))
    assert all(not (r['start_frame']<=10<=r['end_frame']) for r in detect_motion_intervals(df,sample_rate=5))


def test_invalid_sample_rate_fails_before_opening_video():
    with pytest.raises(ValueError,match='sample_rate'):
        analyze_video_stream('not-a-file.mp4',0)


def test_boolean_video_option_and_ranges():
    assert not parse_arguments(['--no-save-video']).save_video
    assert compress_frame_ranges([5,2,1,2,7,6])==[[1,2],[5,7]]


def test_basic_cache_mismatch_is_rejected():
    with pytest.raises(ValueError,match='Cache metadata'):
        validate_video_metadata(dict(width=100),dict(width=200))


@pytest.fixture(scope='module')
def marker_detector():
    return StaffTagDetector(widths=(24,),angle_step=90)


def test_brightness_alone_is_not_marker_evidence(marker_detector):
    assert marker_detector.inspect_crop(np.full((80,80,3),220,np.uint8)).score==0


def test_reference_marker_is_found_after_rotation(marker_detector):
    root=Path(__file__).resolve().parents[1]
    marker=cv2.imread(str(root/'assets/marker_dark.png'))
    marker=np.ascontiguousarray(np.rot90(marker))
    canvas=np.full((100,100,3),35,np.uint8)
    canvas[40:40+marker.shape[0],40:40+marker.shape[1]]=marker
    evidence=marker_detector.inspect_crop(canvas)
    assert evidence.score>.95
    a,b,c,d=evidence.bbox
    assert a<=50<=c and b<=50<=d


def test_consensus_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        ConsensusConfig(marker_threshold=.98,clear_marker_threshold=.90)
