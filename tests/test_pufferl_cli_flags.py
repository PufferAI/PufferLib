from pathlib import Path


def test_eval_gif_flags_are_not_registered_without_capture_support():
    source = Path('pufferlib/pufferl.py').read_text()

    assert '--render-mode' not in source
    assert '--save-frames' not in source
    assert '--gif-path' not in source
    assert '--fps' not in source
