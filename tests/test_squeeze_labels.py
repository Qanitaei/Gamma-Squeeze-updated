from gamma_squeeze.labels.squeeze_definition import (
    is_candidate_setup,
    is_squeeze_outcome,
)


def test_candidate_setup_fragile():
    row = {
        "positioning_stress": 0.5,
        "flag_fragile_gamma": 1.0,
        "flag_near_flip": 0.0,
        "flag_call_crowding": 0.0,
    }
    assert is_candidate_setup(row) is True


def test_candidate_rejects_low_stress():
    row = {
        "positioning_stress": 0.1,
        "flag_fragile_gamma": 1.0,
        "flag_near_flip": 0.0,
        "flag_call_crowding": 0.0,
    }
    assert is_candidate_setup(row) is False


def test_outcome_thresholds():
    assert is_squeeze_outcome(0.03, 1) is True
    assert is_squeeze_outcome(0.02, 1) is False
    assert is_squeeze_outcome(0.09, 10) is True
