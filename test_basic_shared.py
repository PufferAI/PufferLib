from pdb import set_trace as T
import numpy as np
import time

import selectors
from multiprocessing import Process, Pipe, Array, RawArray
from multiprocessing.shared_memory import SharedMemory

def worker_process(envs_per_worker, delay_mean, delay_std, bandwidth,
        send_pipe, recv_pipe, shared_arr):
    data = np.random.randn(bandwidth)
    np_shared = np.frombuffer(shared_arr.get_obj())
    #np_shared = np.frombuffer(shared_arr)
    #np_shared = np.frombuffer(shared_arr.buf)
    #np_shared = shared_arr
    while True:
        request = recv_pipe.recv()
        for _ in range(envs_per_worker):
            start = time.process_time()
            idx = 0
            np_shared[:] = data
            target_time = delay_mean + delay_std*np.random.randn()
            #while time.process_time() - start < target_time:
            while idx < target_time * 49_000_000:
                idx += 1

        send_pipe.send('end')

def test_speed(envs_per_worker=1, delay_mean=0.01, delay_std=0.001,
        bandwidth=1, num_workers=4, batch_size=4, sync=True, timeout=10):
    main_send_pipes, work_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
    work_send_pipes, main_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
    shared_mem = [Array('d', bandwidth) for _ in range(num_workers)]
    #shared_mem = [RawArray('d', bandwidth) for _ in range(num_workers)]
    #shared_mem = [SharedMemory(create=True, size=bandwidth*8) for _ in range(num_workers)]
    #shared_mem = [np.memmap(f'/dev/shm/puf_shared{i}', dtype='float64', mode='w+', shape=(bandwidth,)) for i in range(num_workers)]
    processes = [Process(
        target=worker_process,
        args=(envs_per_worker, delay_mean, delay_std, bandwidth,
        work_send_pipes[i], work_recv_pipes[i], shared_mem[i]))
        for i in range(num_workers)
    ]
    for p in processes:
        p.start()
 

    # Register all receive pipes with the selector
    sel = selectors.DefaultSelector()
    for pipe in main_recv_pipes:
        sel.register(pipe, selectors.EVENT_READ)

    send_idxs = {i for i in range(num_workers)}
    steps_collected = 0
    start = time.time()
    while time.time() - start < timeout:
        for idx in send_idxs:
            main_send_pipes[idx].send('start')

        send_idxs = set()
        if sync:
            for idx, pipe in enumerate(main_recv_pipes):
                assert pipe.recv() == 'end'
                data = shared_mem[idx]
                send_idxs.add(idx)

            steps_collected += num_workers*envs_per_worker
        else:
            while len(send_idxs) < batch_size:
                for key, _ in sel.select(timeout=None):
                    pipe = key.fileobj
                    idx = main_recv_pipes.index(pipe)

                    if pipe.poll():
                        assert pipe.recv() == 'end'
                        data = shared_mem[idx]
                        send_idxs.add(idx)

                    if len(send_idxs) == batch_size:
                        break
                
            steps_collected += batch_size*envs_per_worker

    end = time.time()

    for p in processes:
        p.terminate()

    sps = steps_collected / (end - start)
    print(
        f'SPS: {sps:.2f}',
        f'envs_per_worker: {envs_per_worker}',
        f'delay_mean: {delay_mean}',
        f'delay_std: {delay_std}',
        f'bandwidth: {bandwidth}',
        f'num_workers: {num_workers}',
        f'batch_size: {batch_size}',
        f'sync: {sync}',
    )


if __name__ == '__main__':
    test_speed(delay_mean=0.000, delay_std=0.0000, num_workers=64,
        bandwidth=40, batch_size=24, sync=False, timeout=5)
    exit(0)
    #timeout = 1
    #test_speed(timeout=1)
    test_speed(delay_mean=0, delay_std=0, num_workers=1, batch_size=1, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=1, batch_size=1, sync=True)
    test_speed(delay_mean=0, delay_std=0, num_workers=6, batch_size=6, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=6, batch_size=6, sync=True)
    test_speed(delay_mean=0, delay_std=0, num_workers=24, batch_size=6, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=24, batch_size=24, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=24, batch_size=6, sync=True)

