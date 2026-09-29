import pytest
from src.evaluate import evaluate


def test_unlabelled_frames_are_not_counted_as_negatives():
    result=evaluate([1,3,999],[{'frame_id':1,'staff_present':True},
                              {'frame_id':2,'staff_present':True},
                              {'frame_id':3,'staff_present':False},
                              {'frame_id':4,'staff_present':None}])
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert result['labelled_frames']==3
    assert result['precision']==.5 and result['recall']==.5


def test_localization_counts_misses_and_false_predictions():
    result=evaluate([1],[{'frame_id':1,'staff_present':True,'points':[[0,0],[100,100]]}],
                    {1:[[3,4],[500,500]]},match_radius=10)
    loc=result['localization']
    assert loc['matched']==1 and loc['missed']==1 and loc['false_positives']==1
    assert loc['mean_error_pixels']==5


def test_duplicate_labels_fail():
    with pytest.raises(ValueError):
        evaluate([],[{'frame_id':0,'staff_present':False}]*2)
