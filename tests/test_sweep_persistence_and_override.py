"""Tests for Protein sweep persistence and override.
Run: python tests/test_sweep_persistence_and_override.py

Tests cover:
- State persistence (save/load, crash recovery, atomic writes)
- Override injection (single/multiple, partial params, consumption)
- Analysis helpers (read_sweep_results, create_override)
"""
import os
import json
import tempfile
import numpy as np


def _minimal_sweep_config():
    """Minimal config for testing."""
    return {
        'method': 'Protein',
        'metric': 'score',
        'goal': 'maximize',
        'downsample': 1,
        'train': {
            'learning_rate': {
                'distribution': 'log_normal',
                'min': 0.0001, 'max': 0.01, 'mean': 0.001, 'scale': 0.5
            },
            'total_timesteps': {
                'distribution': 'log_normal',
                'min': 1e7, 'max': 1e9, 'mean': 1e8, 'scale': 'time'
            },
        }
    }


# =============================================================================
# Persistence Tests
# =============================================================================

def test_json_default():
    from pufferlib.sweep import Protein
    arr = np.array([1.0, 2.0])
    assert Protein._json_default(arr) == [1.0, 2.0]
    assert Protein._json_default(np.float64(1.5)) == 1.5
    print("PASS test_json_default")


def test_save_and_load_state():
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        p1 = Protein(cfg, use_gpu=False)
        p1.success_observations = [{'input': np.array([0.1, 0.2]), 'output': 0.8, 'cost': 100}]
        p1.suggestion_idx = 5
        p1._save_state()

        cfg2 = _minimal_sweep_config()
        cfg2['state_file'] = os.path.join(tmpdir, "test.json")
        cfg2['override_file'] = os.path.join(tmpdir, "int2.json")
        p2 = Protein(cfg2, use_gpu=False)
        assert p2.suggestion_idx == 5
        assert len(p2.success_observations) == 1
        print("PASS test_save_and_load_state")


def test_override():
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['override_file'] = os.path.join(tmpdir, "int.json")
        cfg['state_file'] = os.path.join(tmpdir, "state.json")

        # Create override with 2 suggestions
        with open(cfg['override_file'], 'w') as f:
            json.dump({'suggestions': [
                {'params': {'train/learning_rate': 0.005}, 'reason': 'test1'},
                {'params': {'train/learning_rate': 0.006}, 'reason': 'test2'},
            ]}, f)

        p = Protein(cfg, use_gpu=False)

        # First call consumes first suggestion
        result = p._check_override()
        assert result == {'train/learning_rate': 0.005}
        assert os.path.exists(cfg['override_file'])  # still has one left

        # Second call consumes second and deletes file
        result = p._check_override()
        assert result == {'train/learning_rate': 0.006}
        assert not os.path.exists(cfg['override_file'])  # consumed
        print("PASS test_override")


def test_override_in_suggest():
    """Test that override params are used in suggest()."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['override_file'] = os.path.join(tmpdir, "int.json")
        cfg['state_file'] = os.path.join(tmpdir, "state.json")

        # Create override
        with open(cfg['override_file'], 'w') as f:
            json.dump({'suggestions': [
                {'params': {'train/learning_rate': 0.005, 'train/total_timesteps': 5e7}, 'reason': 'test'},
            ]}, f)

        p = Protein(cfg, use_gpu=False)
        fill = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}}
        result, info = p.suggest(fill)

        assert info.get('override') is True
        assert abs(result['train']['learning_rate'] - 0.005) < 1e-6
        assert not os.path.exists(cfg['override_file'])  # consumed
        print("PASS test_override_in_suggest")


def test_atomic_write():
    """Test that _save_state uses atomic write."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        p = Protein(cfg, use_gpu=False)
        p.success_observations = [{'input': np.array([0.1, 0.2]), 'output': 0.8, 'cost': 100}]
        p._save_state()

        # Verify file exists and is valid JSON
        assert os.path.exists(cfg['state_file'])
        with open(cfg['state_file']) as f:
            state = json.load(f)
        assert len(state['success_observations']) == 1
        print("PASS test_atomic_write")


def test_partial_override():
    """Test that override with only some params merges with fill dict."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['override_file'] = os.path.join(tmpdir, "int.json")
        cfg['state_file'] = os.path.join(tmpdir, "state.json")

        # Create override with only learning_rate (not total_timesteps)
        with open(cfg['override_file'], 'w') as f:
            json.dump({'suggestions': [
                {'params': {'train/learning_rate': 0.0069}, 'reason': 'partial test'},
            ]}, f)

        p = Protein(cfg, use_gpu=False)
        # Fill has both params with default values
        fill = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8, 'extra_param': 42}}
        result, info = p.suggest(fill)

        assert info.get('override') is True
        # Override param should be overwritten
        assert abs(result['train']['learning_rate'] - 0.0069) < 1e-9
        # Non-override param should be preserved from fill
        assert result['train']['total_timesteps'] == 1e8
        assert result['train']['extra_param'] == 42
        # CRITICAL: fill must be modified in place (pufferl.py expects this)
        assert fill is result, "suggest() must modify fill in place, not return a copy"
        assert abs(fill['train']['learning_rate'] - 0.0069) < 1e-9
        print("PASS test_partial_override")


def test_observe_saves_state():
    """Test that observe() automatically saves state."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        p = Protein(cfg, use_gpu=False)
        fill = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}}

        # Get first suggestion
        result, _ = p.suggest(fill)

        # Observe result
        p.observe(result, score=0.75, cost=100)

        # Check state was saved
        assert os.path.exists(cfg['state_file'])
        with open(cfg['state_file']) as f:
            state = json.load(f)
        assert len(state['success_observations']) == 1
        assert state['success_observations'][0]['output'] == 0.75
        print("PASS test_observe_saves_state")


def test_failure_observation():
    """Test that failure observations are recorded."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        p = Protein(cfg, use_gpu=False)
        fill = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}}
        result, _ = p.suggest(fill)

        # Observe failure
        p.observe(result, score=float('nan'), cost=100)

        with open(cfg['state_file']) as f:
            state = json.load(f)
        assert len(state['failure_observations']) == 1
        assert state['failure_observations'][0]['is_failure'] is True
        print("PASS test_failure_observation")


def test_crash_recovery_preserves_bounds():
    """Test that min/max score bounds are preserved across crash recovery."""
    from pufferlib.sweep import Protein
    import math
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        # First session - add observations
        p1 = Protein(cfg, use_gpu=False)
        p1.success_observations = [
            {'input': np.array([0.1, 0.2]), 'output': 0.3, 'cost': 50},
            {'input': np.array([0.2, 0.3]), 'output': 0.9, 'cost': 100},
        ]
        p1.min_score = 0.3
        p1.max_score = 0.9
        p1.log_c_min = np.log(50)
        p1.log_c_max = np.log(100)
        p1._save_state()

        # Second session - recover
        cfg2 = _minimal_sweep_config()
        cfg2['state_file'] = os.path.join(tmpdir, "test.json")
        cfg2['override_file'] = os.path.join(tmpdir, "int2.json")
        p2 = Protein(cfg2, use_gpu=False)

        assert p2.min_score == 0.3
        assert p2.max_score == 0.9
        assert abs(p2.log_c_min - np.log(50)) < 1e-9
        assert abs(p2.log_c_max - np.log(100)) < 1e-9
        print("PASS test_crash_recovery_preserves_bounds")


def test_invalid_override_file():
    """Test that invalid override files are handled gracefully."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['override_file'] = os.path.join(tmpdir, "int.json")
        cfg['state_file'] = os.path.join(tmpdir, "state.json")

        # Create invalid JSON
        with open(cfg['override_file'], 'w') as f:
            f.write("not valid json {{{")

        p = Protein(cfg, use_gpu=False)
        result = p._check_override()

        # Should return None and delete the invalid file
        assert result is None
        assert not os.path.exists(cfg['override_file'])
        print("PASS test_invalid_override_file")


def test_sweep_continues_after_crash():
    """Test that a sweep can be stopped and resumed from where it left off.

    Simulates: run 2 iterations -> crash -> resume -> run 2 more iterations
    Verifies: suggestion_idx continues, observations accumulate, no duplicates,
              AND that loaded observations are numpy arrays with correct values.
    """
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "sweep.json")
        int_file = os.path.join(tmpdir, "int.json")

        # --- Session 1: Run 2 iterations then "crash" ---
        cfg1 = _minimal_sweep_config()
        cfg1['state_file'] = state_file
        cfg1['override_file'] = int_file

        p1 = Protein(cfg1, use_gpu=False)
        fill = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}}

        # Iteration 1
        result1, _ = p1.suggest(fill.copy())
        p1.observe(result1, score=0.5, cost=100)

        # Iteration 2
        result2, _ = p1.suggest(fill.copy())
        p1.observe(result2, score=0.6, cost=150)

        assert p1.suggestion_idx == 2
        assert len(p1.success_observations) == 2

        # Capture state for deep verification after resume
        p1_inputs = [obs['input'].copy() for obs in p1.success_observations]
        p1_outputs = [obs['output'] for obs in p1.success_observations]
        p1_costs = [obs['cost'] for obs in p1.success_observations]
        p1_min_score = p1.min_score
        p1_max_score = p1.max_score
        p1_log_c_min = p1.log_c_min
        p1_log_c_max = p1.log_c_max

        # Verify state file exists
        assert os.path.exists(state_file)

        # --- "Crash" - delete p1, create new instance ---
        del p1

        # --- Session 2: Resume and run 2 more iterations ---
        cfg2 = _minimal_sweep_config()
        cfg2['state_file'] = state_file
        cfg2['override_file'] = int_file

        p2 = Protein(cfg2, use_gpu=False)

        # Verify state was recovered
        assert p2.suggestion_idx == 2, f"Expected idx=2, got {p2.suggestion_idx}"
        assert len(p2.success_observations) == 2, f"Expected 2 obs, got {len(p2.success_observations)}"

        # Deep verification: observations are numpy arrays (not JSON lists)
        for i, obs in enumerate(p2.success_observations):
            assert isinstance(obs['input'], np.ndarray), \
                f"Obs {i} input should be np.ndarray, got {type(obs['input'])}"

        # Deep verification: observation values match exactly
        for i, obs in enumerate(p2.success_observations):
            assert np.allclose(obs['input'], p1_inputs[i]), f"Obs {i} input mismatch"
            assert obs['output'] == p1_outputs[i], f"Obs {i} output mismatch"
            assert obs['cost'] == p1_costs[i], f"Obs {i} cost mismatch"

        # Deep verification: bounds restored correctly (handle inf case)
        import math
        if math.isinf(p1_min_score):
            assert math.isinf(p2.min_score), f"min_score: expected inf, got {p2.min_score}"
        else:
            assert p2.min_score == p1_min_score, f"min_score mismatch"
        if math.isinf(p1_max_score):
            assert math.isinf(p2.max_score), f"max_score: expected -inf, got {p2.max_score}"
        else:
            assert p2.max_score == p1_max_score, f"max_score mismatch"
        if math.isinf(p1_log_c_min):
            assert math.isinf(p2.log_c_min), f"log_c_min: expected inf, got {p2.log_c_min}"
        else:
            assert abs(p2.log_c_min - p1_log_c_min) < 1e-9, f"log_c_min mismatch"
        if math.isinf(p1_log_c_max):
            assert math.isinf(p2.log_c_max), f"log_c_max: expected -inf, got {p2.log_c_max}"
        else:
            assert abs(p2.log_c_max - p1_log_c_max) < 1e-9, f"log_c_max mismatch"

        # Iteration 3
        result3, _ = p2.suggest(fill.copy())
        p2.observe(result3, score=0.7, cost=200)

        # Iteration 4
        result4, _ = p2.suggest(fill.copy())
        p2.observe(result4, score=0.8, cost=250)

        assert p2.suggestion_idx == 4
        assert len(p2.success_observations) == 4

        # Verify all scores are present
        scores = [obs['output'] for obs in p2.success_observations]
        assert 0.5 in scores
        assert 0.6 in scores
        assert 0.7 in scores
        assert 0.8 in scores

        print("PASS test_sweep_continues_after_crash")


def test_corrupted_state_file():
    """Truncated/corrupted JSON should reset state, not crash."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        # Write truncated JSON (missing closing brace)
        with open(cfg['state_file'], 'w') as f:
            f.write('{"suggestion_idx": 5')

        # Should not crash - should start fresh
        p = Protein(cfg, use_gpu=False)
        assert p.suggestion_idx == 0
        assert len(p.success_observations) == 0
        print("PASS test_corrupted_state_file")


def test_empty_state_file():
    """Empty JSON object {} should use defaults, not crash."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        # Write empty JSON object
        with open(cfg['state_file'], 'w') as f:
            f.write('{}')

        # Should use defaults
        p = Protein(cfg, use_gpu=False)
        assert p.suggestion_idx == 0
        assert len(p.success_observations) == 0
        print("PASS test_empty_state_file")


def test_state_file_deleted_during_load():
    """FileNotFoundError during load should not crash."""
    from pufferlib.sweep import Protein
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        # Patch exists() to return True even though file doesn't exist
        # This simulates the race condition
        original_exists = os.path.exists
        def mock_exists(path):
            if path == cfg['state_file']:
                return True
            return original_exists(path)

        with patch('os.path.exists', side_effect=mock_exists):
            p = Protein(cfg, use_gpu=False)

        assert p.suggestion_idx == 0  # Started fresh despite "existing" file
        print("PASS test_state_file_deleted_during_load")


def test_save_state_cleans_up_tmp_on_failure():
    """_save_state() should clean up .tmp file if write fails."""
    from pufferlib.sweep import Protein
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        p = Protein(cfg, use_gpu=False)
        tmp_file = f"{cfg['state_file']}.tmp"

        # Patch os.replace to fail
        with patch('os.replace', side_effect=OSError("Simulated disk full")):
            p._save_state()

        # .tmp file should NOT exist (cleaned up)
        assert not os.path.exists(tmp_file), ".tmp file should be cleaned up on failure"
        print("PASS test_save_state_cleans_up_tmp_on_failure")


def test_orphaned_tmp_cleaned_on_load():
    """Orphaned .tmp files from previous crash should be cleaned up on load."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['state_file'] = os.path.join(tmpdir, "test.json")
        cfg['override_file'] = os.path.join(tmpdir, "int.json")

        tmp_file = f"{cfg['state_file']}.tmp"

        # Create orphaned .tmp file (simulates crash during previous save)
        with open(tmp_file, 'w') as f:
            f.write('orphaned tmp data')

        assert os.path.exists(tmp_file)

        # Initialize Protein - should clean up orphan
        p = Protein(cfg, use_gpu=False)

        assert not os.path.exists(tmp_file), "Orphaned .tmp should be cleaned up"
        print("PASS test_orphaned_tmp_cleaned_on_load")


def test_override_invalid_path():
    """Override with non-existent nested path should skip, not crash."""
    from pufferlib.sweep import Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['override_file'] = os.path.join(tmpdir, "int.json")
        cfg['state_file'] = os.path.join(tmpdir, "state.json")

        # Override has valid keys and invalid path
        with open(cfg['override_file'], 'w') as f:
            json.dump({'suggestions': [{
                'params': {
                    'train/learning_rate': 0.005,
                    'nonexistent/deeply/nested': 0.2,
                },
                'reason': 'test invalid path'
            }]}, f)

        p = Protein(cfg, use_gpu=False)
        fill = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}}

        # Should not crash, should apply valid keys, skip invalid paths
        result, info = p.suggest(fill)

        assert info.get('override') is True
        assert abs(result['train']['learning_rate'] - 0.005) < 1e-6
        assert 'nonexistent' not in result
        print("PASS test_override_invalid_path")


def test_override_atomic_write():
    """Override update should use atomic write pattern."""
    from pufferlib.sweep import Protein
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _minimal_sweep_config()
        cfg['override_file'] = os.path.join(tmpdir, "int.json")
        cfg['state_file'] = os.path.join(tmpdir, "state.json")
        tmp_file = f"{cfg['override_file']}.tmp"

        # Override with 2 suggestions
        with open(cfg['override_file'], 'w') as f:
            json.dump({'suggestions': [
                {'params': {'train/learning_rate': 0.005}, 'reason': 'first'},
                {'params': {'train/learning_rate': 0.006}, 'reason': 'second'},
            ]}, f)

        # Patch os.replace to fail (simulating crash during atomic write)
        p = Protein(cfg, use_gpu=False)

        with patch('os.replace', side_effect=OSError("Simulated crash")):
            result = p._check_override()

        # Original file should still be intact (no corruption)
        with open(cfg['override_file']) as f:
            data = json.load(f)

        # Should still have both suggestions (first wasn't consumed due to crash)
        assert len(data['suggestions']) == 2, "Original file should be intact after crash"
        # .tmp should be cleaned up
        assert not os.path.exists(tmp_file), ".tmp should be cleaned up on failure"
        print("PASS test_override_atomic_write")


# =============================================================================
# Analysis Helper Tests
# =============================================================================

def test_read_sweep_results():
    """Read state file and return denormalized observations."""
    from pufferlib.sweep import read_sweep_results, Hyperparameters
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, 'test_sweep.json')
        config = _minimal_sweep_config()

        # Create Hyperparameters to get normalized values
        hyperparams = Hyperparameters(config, verbose=False)

        # Create a known observation with specific real values
        real_params = {'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}}
        normalized = hyperparams.from_dict(real_params)

        # Write state file with this observation
        state = {
            'success_observations': [
                {'input': normalized.tolist(), 'output': 0.75, 'cost': 100.0}
            ],
            'failure_observations': []
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Read and denormalize
        results = read_sweep_results(state_file, config)

        assert len(results) == 1
        assert results[0]['score'] == 0.75
        assert results[0]['cost'] == 100.0

        # Check params are denormalized and flattened
        params = results[0]['params']
        assert 'train/learning_rate' in params
        assert 'train/total_timesteps' in params

        # Values should be close to originals (some numerical precision loss)
        assert abs(params['train/learning_rate'] - 0.001) < 1e-6
        assert abs(params['train/total_timesteps'] - 1e8) / 1e8 < 0.01

    print("PASS test_read_sweep_results")


def test_read_sweep_results_sorted():
    """Results sorted by score (descending) by default."""
    from pufferlib.sweep import read_sweep_results, Hyperparameters
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, 'test_sweep.json')
        config = _minimal_sweep_config()

        hyperparams = Hyperparameters(config, verbose=False)
        normalized = hyperparams.from_dict({'train': {'learning_rate': 0.001, 'total_timesteps': 1e8}})

        # Create observations with different scores
        state = {
            'success_observations': [
                {'input': normalized.tolist(), 'output': 0.5, 'cost': 100.0},
                {'input': normalized.tolist(), 'output': 0.9, 'cost': 200.0},
                {'input': normalized.tolist(), 'output': 0.3, 'cost': 50.0},
            ],
            'failure_observations': []
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Default sort by score descending
        results = read_sweep_results(state_file, config)
        assert results[0]['score'] == 0.9
        assert results[1]['score'] == 0.5
        assert results[2]['score'] == 0.3

        # Sort by cost ascending
        results = read_sweep_results(state_file, config, sort_by='cost')
        assert results[0]['cost'] == 50.0
        assert results[1]['cost'] == 100.0
        assert results[2]['cost'] == 200.0

        # No sort
        results = read_sweep_results(state_file, config, sort_by=None)
        assert results[0]['score'] == 0.5  # Original order

    print("PASS test_read_sweep_results_sorted")


def test_read_sweep_results_empty():
    """Empty state file returns empty list."""
    from pufferlib.sweep import read_sweep_results
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, 'test_sweep.json')
        config = _minimal_sweep_config()

        # Empty state
        state = {'success_observations': [], 'failure_observations': []}
        with open(state_file, 'w') as f:
            json.dump(state, f)

        results = read_sweep_results(state_file, config)
        assert results == []

    print("PASS test_read_sweep_results_empty")


def test_config_mismatch_error():
    """Clear error when state file doesn't match config dimensions."""
    from pufferlib.sweep import read_sweep_results
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, 'test_sweep.json')

        # Config with 2 params
        config = _minimal_sweep_config()

        # State with 3 dimensions (wrong!)
        state = {
            'success_observations': [
                {'input': [0.1, 0.2, 0.3], 'output': 0.75, 'cost': 100.0}
            ],
            'failure_observations': []
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

        # Should raise ValueError with helpful message
        try:
            read_sweep_results(state_file, config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "3 dimensions" in str(e)
            assert "2" in str(e)
            assert "config" in str(e).lower()

    print("PASS test_config_mismatch_error")


def test_create_override_single():
    """Create override file with one suggestion."""
    from pufferlib.sweep import create_override, Protein
    with tempfile.TemporaryDirectory() as tmpdir:
        int_file = os.path.join(tmpdir, 'override.json')

        # Create override
        create_override(int_file, [{'train/learning_rate': 0.00069}], reason=['test single'])

        # Verify file format
        with open(int_file) as f:
            data = json.load(f)

        assert 'suggestions' in data
        assert len(data['suggestions']) == 1
        assert data['suggestions'][0]['params'] == {'train/learning_rate': 0.00069}
        assert data['suggestions'][0]['reason'] == 'test single'

        # Verify it can be consumed by Protein._check_override()
        config = _minimal_sweep_config()
        config['override_file'] = int_file
        config['state_file'] = os.path.join(tmpdir, 'state.json')
        p = Protein(config, use_gpu=False)
        result = p._check_override()
        assert result == {'train/learning_rate': 0.00069}
        assert not os.path.exists(int_file)  # Consumed and deleted

    print("PASS test_create_override_single")


def test_create_override_multiple():
    """Create override with multiple suggestions."""
    from pufferlib.sweep import create_override
    with tempfile.TemporaryDirectory() as tmpdir:
        int_file = os.path.join(tmpdir, 'override.json')

        # Multiple suggestions with different reasons
        suggestions = [
            {'train/learning_rate': 0.001},
            {'train/learning_rate': 0.002},
            {'train/learning_rate': 0.003},
        ]
        reasons = ['reason1', 'reason2', 'reason3']

        create_override(int_file, suggestions, reason=reasons)

        with open(int_file) as f:
            data = json.load(f)

        assert len(data['suggestions']) == 3
        assert data['suggestions'][0]['params'] == {'train/learning_rate': 0.001}
        assert data['suggestions'][0]['reason'] == 'reason1'
        assert data['suggestions'][1]['params'] == {'train/learning_rate': 0.002}
        assert data['suggestions'][1]['reason'] == 'reason2'
        assert data['suggestions'][2]['params'] == {'train/learning_rate': 0.003}
        assert data['suggestions'][2]['reason'] == 'reason3'

    print("PASS test_create_override_multiple")


def test_create_override_reasons():
    """Reasons list must match suggestions length, or be None."""
    from pufferlib.sweep import create_override
    with tempfile.TemporaryDirectory() as tmpdir:
        # Reasons list matches suggestions
        int_file = os.path.join(tmpdir, 'int1.json')
        create_override(int_file, [{'a': 1}, {'b': 2}], reason=['reason1', 'reason2'])
        with open(int_file) as f:
            data = json.load(f)
        assert data['suggestions'][0]['reason'] == 'reason1'
        assert data['suggestions'][1]['reason'] == 'reason2'

        # No reason - should use default
        int_file = os.path.join(tmpdir, 'int2.json')
        create_override(int_file, [{'a': 1}])
        with open(int_file) as f:
            data = json.load(f)
        assert 'reason' in data['suggestions'][0]
        assert data['suggestions'][0]['reason']  # Non-empty default

    print("PASS test_create_override_reasons")


def test_create_override_mismatched_reasons():
    """Mismatched list lengths raise ValueError."""
    from pufferlib.sweep import create_override
    with tempfile.TemporaryDirectory() as tmpdir:
        int_file = os.path.join(tmpdir, 'int.json')
        try:
            create_override(int_file, [{'a': 1}, {'b': 2}], reason=['only one'])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "2 suggestions" in str(e)
            assert "1 reasons" in str(e)

    print("PASS test_create_override_mismatched_reasons")


if __name__ == '__main__':
    # Persistence tests
    test_json_default()
    test_save_and_load_state()
    test_override()
    test_override_in_suggest()
    test_atomic_write()
    test_partial_override()
    test_observe_saves_state()
    test_failure_observation()
    test_crash_recovery_preserves_bounds()
    test_invalid_override_file()
    test_sweep_continues_after_crash()
    test_corrupted_state_file()
    test_empty_state_file()
    test_state_file_deleted_during_load()
    test_save_state_cleans_up_tmp_on_failure()
    test_orphaned_tmp_cleaned_on_load()
    test_override_invalid_path()
    test_override_atomic_write()
    # Analysis helper tests
    test_read_sweep_results()
    test_read_sweep_results_sorted()
    test_read_sweep_results_empty()
    test_config_mismatch_error()
    test_create_override_single()
    test_create_override_multiple()
    test_create_override_reasons()
    test_create_override_mismatched_reasons()
    print("\nOK: All 26 tests passed!")
