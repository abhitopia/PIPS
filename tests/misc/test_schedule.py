import pytest
import numpy as np
from pips.misc.schedule import Schedule

def test_linear_schedule():
    schedule_fn = Schedule.get_schedule(
        initial_value=0.0,
        target_value=1.0,
        warmup_steps=10,
        schedule_type='linear'
    )
    
    # Test initial value
    assert schedule_fn(0) == 0.0
    
    # Test midpoint
    assert np.isclose(schedule_fn(5), 0.5)
    
    # Test final value
    assert schedule_fn(10) == 1.0
    
    # Test after warmup
    assert schedule_fn(15) == 1.0

def test_exponential_schedule():
    schedule_fn = Schedule.get_schedule(
        initial_value=1.0,
        target_value=0.1,
        warmup_steps=10,
        schedule_type='exponential'
    )
    
    # Test initial value
    assert schedule_fn(0) == 1.0
    
    # Test that values decrease exponentially
    assert schedule_fn(5) > schedule_fn(7)
    
    # Test final value
    assert np.isclose(schedule_fn(10), 0.1, rtol=1e-3)
    
    # Test after warmup
    assert np.isclose(schedule_fn(15), 0.1, rtol=1e-3)

def test_cosine_decay_schedule():
    schedule_fn = Schedule.get_schedule(
        initial_value=1.0,
        target_value=0.0,
        warmup_steps=10,
        schedule_type='cosine_decay'
    )
    
    # Test initial value
    assert np.isclose(schedule_fn(0), 1.0)
    
    # Test midpoint
    assert np.isclose(schedule_fn(5), 0.5)
    
    # Test final value
    assert np.isclose(schedule_fn(10), 0.0)
    
    # Test after warmup
    assert np.isclose(schedule_fn(15), 0.0)

def test_cosine_anneal_schedule():
    schedule_fn = Schedule.get_schedule(
        initial_value=0.0,
        target_value=1.0,
        warmup_steps=10,
        schedule_type='cosine_anneal'
    )
    
    # Test initial value
    assert np.isclose(schedule_fn(0), 0.0)
    
    # Test midpoint
    assert np.isclose(schedule_fn(5), 0.5)
    
    # Test final value
    assert np.isclose(schedule_fn(10), 1.0)
    
    # Test after warmup
    assert np.isclose(schedule_fn(15), 1.0)

def test_threshold_schedule():
    schedule_fn = Schedule.get_schedule(
        initial_value=0.0,
        target_value=1.0,
        warmup_steps=10,
        schedule_type='threshold'
    )
    
    # Test initial value
    assert schedule_fn(0) == 0.0
    
    # Test just before threshold
    assert schedule_fn(9) == 0.0
    
    # Test at threshold
    assert schedule_fn(10) == 1.0
    
    # Test after threshold
    assert schedule_fn(15) == 1.0

def test_invalid_schedule_type():
    with pytest.raises(ValueError, match="Unknown schedule type: invalid"):
        Schedule.get_schedule(
            initial_value=0.0,
            target_value=1.0,
            warmup_steps=10,
            schedule_type='invalid'
        )

def test_zero_warmup_steps():
    schedule_fn = Schedule.get_schedule(
        initial_value=0.0,
        target_value=1.0,
        warmup_steps=0,
        schedule_type='linear'
    )
    
    # Should return target value immediately
    assert schedule_fn(0) == 1.0
    assert schedule_fn(1) == 1.0

def test_negative_values():
    schedule_fn = Schedule.get_schedule(
        initial_value=-1.0,
        target_value=-0.1,
        warmup_steps=10,
        schedule_type='linear'
    )
    
    # Test initial value
    assert schedule_fn(0) == -1.0
    
    # Test midpoint
    assert np.isclose(schedule_fn(5), -0.55)
    
    # Test final value
    assert schedule_fn(10) == -0.1 