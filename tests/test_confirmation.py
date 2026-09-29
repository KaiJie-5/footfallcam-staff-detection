"""Regression cases for reviewed badge matches and misleading background patterns."""
from src.temporal import classify_tracks
from src.pipeline_runner import export_results, parse_arguments
from src.detector import CorridorZoneFilter


def sample(frame, score, person, badge, template='marker_light.png'):
    x, y, x2, y2 = person
    return dict(frame_id=frame, quality=dict(valid=True, reason='ok'), detections=[
        dict(track_id=1, bbox=person, center=[(x+x2)/2, (y+y2)/2], conf=.8,
             badge=dict(score=score, bbox=badge, contrast=50., angle=0, template=template))])


def test_reviewed_clear_match_does_not_need_a_second_strong_hit():
    # Score and boxes from the reviewed frame-860 cache observation.
    frames = [sample(860, .99059, [498,95,605,257], [557,167,581,185], 'marker_dark.png')]
    classify_tracks(frames, 25)
    d = frames[0]['detections'][0]
    assert d['status'] == 'staff'
    assert d['reason'] == 'clear_marker'


def test_reviewed_light_marker_matches_support_one_another():
    # Scores and boxes from the existing cache, including the truncated entry box.
    frames = [sample(425, .83730, [478,572,557,720], [496,687,530,712]),
              sample(430, .87902, [480,508,568,703], [508,619,546,640]),
              sample(435, .82186, [507,444,627,600], [558,513,585,532])]
    classify_tracks(frames, 25)
    assert all(f['detections'][0]['status'] == 'staff' for f in frames)
    assert frames[1]['detections'][0]['reason'] == 'consistent_marker'


def test_a_fixed_background_pattern_is_not_confirmed_by_repetition():
    # A real rejected background match scores higher than the frame-430 badge.
    frames = [sample(50, .87932, [634,591,755,668], [678,606,690,622], 'marker_dark.png'),
              sample(55, .87914, [634,594,754,668], [678,606,690,622], 'marker_dark.png')]
    classify_tracks(frames, 25)
    assert all(f['detections'][0]['status'] != 'staff' for f in frames)


def test_moving_person_does_not_confirm_a_fixed_patch_in_the_box():
    frames = [sample(0, .89, [0,0,120,200], [60,80,80,95]),
              sample(5, .88, [40,0,160,200], [60,80,80,95])]
    classify_tracks(frames, 25)
    assert all(f['detections'][0]['status'] != 'staff' for f in frames)


def test_single_moderate_hit_stays_a_candidate():
    frames = [sample(0, .879, [0,0,120,200], [60,80,80,95])]
    classify_tracks(frames, 25)
    assert frames[0]['detections'][0]['status'] == 'candidate'


def test_clear_score_without_a_badge_inside_the_person_is_rejected():
    frames = [sample(0, .999, [0,0,120,200], [130,80,150,95])]
    classify_tracks(frames, 25)
    assert frames[0]['detections'][0]['status'] != 'staff'


def test_different_track_ids_do_not_share_marker_confirmation():
    frames = [sample(425, .83730, [478,572,557,720], [496,687,530,712]),
              sample(430, .87902, [480,508,568,703], [508,619,546,640])]
    frames[1]['detections'][0]['track_id'] = 2
    classify_tracks(frames, 25)
    assert all(f['detections'][0]['status'] != 'staff' for f in frames)


def test_large_gap_does_not_combine_weaker_matches():
    frames = [sample(0, .88, [0,0,120,200], [60,80,80,95]),
              sample(100, .85, [40,0,160,200], [100,80,120,95])]
    classify_tracks(frames, 25)
    assert all(f['detections'][0]['status'] != 'staff' for f in frames)


def test_detailed_frame_log_is_opt_in(tmp_path):
    frames = [sample(0, .999, [0,0,120,200], [60,80,80,95])]
    classify_tracks(frames, 25)
    metadata = dict(width=200, height=240, fps=25)
    export_results(frames, metadata, tmp_path, CorridorZoneFilter())
    assert not (tmp_path/'frame_decisions.jsonl').exists()
    export_results(frames, metadata, tmp_path, CorridorZoneFilter(), debug=True)
    assert (tmp_path/'frame_decisions.jsonl').exists()
    assert parse_arguments(['--debug']).debug
