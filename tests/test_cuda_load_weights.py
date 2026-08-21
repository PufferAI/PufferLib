from pathlib import Path


def test_bf16_load_weights_syncs_cast_before_return():
    source = Path('src/bindings.cu').read_text()
    body = source.split('void load_weights', 1)[1].split('\n}\n\nint py_add_frozen_bank', 1)[0]

    cast_idx = body.index('cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl.default_stream>>>')
    launch_check_idx = body.index('cudaGetLastError()')
    sync_idx = body.index('cudaStreamSynchronize(pufferl.default_stream)')

    assert cast_idx < launch_check_idx < sync_idx
    assert 'throw_if_cuda_error' in body
