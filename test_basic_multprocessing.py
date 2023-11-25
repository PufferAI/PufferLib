from pdb import set_trace as T
import numpy as np
import time

import selectors
from multiprocessing import Process, Pipe

def worker_process(envs_per_worker, delay_mean, delay_std, send_pipe, recv_pipe):
    while True:
        request = recv_pipe.recv()
        for _ in range(envs_per_worker):
            start = time.process_time()
            idx = 0
            target_time = delay_mean + delay_std*np.random.randn()
            while time.process_time() - start < target_time:
                idx += 1

        send_pipe.send('end')

def test_speed(envs_per_worker=1, delay_mean=0.01, delay_std=0.001, num_workers=4, batch_size=4, sync=True, timeout=10):
    main_send_pipes, work_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
    work_send_pipes, main_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])

    processes = [Process(
        target=worker_process,
        args=(envs_per_worker, delay_mean, delay_std, work_send_pipes[i], work_recv_pipes[i]))
        for i in range(num_workers)]

    for p in processes:
        p.start()
 
    send_idxs = {i for i in range(num_workers)}

    # Register all receive pipes with the selector
    sel = selectors.DefaultSelector()
    for pipe in main_recv_pipes:
        sel.register(pipe, selectors.EVENT_READ)

    steps_collected = 0
    start = time.time()
    while time.time() - start < timeout:
        for idx in send_idxs:
            main_send_pipes[idx].send('start')

        send_idxs = set()

        if sync:
            for idx, pipe in enumerate(main_recv_pipes):
                assert pipe.recv() == 'end'
                send_idxs.add(idx)

            steps_collected += num_workers*envs_per_worker
        else:
            for key, _ in sel.select(timeout=None):
                pipe = key.fileobj
                idx = main_recv_pipes.index(pipe)

                if pipe.poll():
                    assert pipe.recv() == 'end'
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
        f'num_workers: {num_workers}',
        f'batch_size: {batch_size}',
        f'sync: {sync}',
    )


if __name__ == '__main__':
    #timeout = 1
    #test_speed(timeout=1)
    test_speed(delay_mean=0, delay_std=0, num_workers=1, batch_size=1, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=1, batch_size=1, sync=True)
    test_speed(delay_mean=0, delay_std=0, num_workers=6, batch_size=6, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=6, batch_size=6, sync=True)
    test_speed(delay_mean=0, delay_std=0, num_workers=24, batch_size=6, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=24, batch_size=24, sync=False)
    test_speed(delay_mean=0, delay_std=0, num_workers=24, batch_size=6, sync=True)

