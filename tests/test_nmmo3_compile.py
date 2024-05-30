from pdb import set_trace as T
import time
import torch
import numpy as np


@torch.compile(fullgraph=True, mode='reduce-overhead')
def fast_decode_map(codes, obs, factors, add, div):
    codes = codes.view(codes.shape[0], 1, -1)
    dec = add + (codes//div) % factors
    obs.scatter_(1, dec, 1)
    return obs

#@torch.compile(fullgraph=True, mode='reduce-overhead')
def decode_map(codes):
    codes = codes.unsqueeze(1).long()
    factors = [4, 4, 16, 5, 3, 5, 5, 6, 7, 4]
    n_channels = sum(factors)
    obs = torch.zeros(codes.shape[0], n_channels, 11, 15, device='cuda')

    add, div = 0, 1
    # TODO: check item/tier order
    for mod in factors:
        obs.scatter_(1, add+(codes//div)%mod, 1)
        add += mod
        div *= mod

    return obs


def test_perf(n=100, agents=1024):
    factors = np.array([4, 4, 16, 5, 3, 5, 5, 6, 7, 4])
    n_channels = sum(factors)
    add = np.array([0, *np.cumsum(factors).tolist()[:-1]])[None, :, None]
    div = np.array([1, *np.cumprod(factors).tolist()[:-1]])[None, :, None]

    factors = torch.tensor(factors)[None, :, None].cuda()
    add = torch.tensor(add).cuda()
    div = torch.tensor(div).cuda()

    codes = torch.randint(0, 4*4*16*5*3*5*5*6*7*4, (agents, 11, 15)).cuda()
    obs = torch.zeros(agents, n_channels, 11*15, device='cuda')
    obs_view = obs.view(agents, n_channels, 11, 15)

    # Warm up
    decode_map(codes)
    fast_decode_map(codes, obs, factors, add, div)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n):
        fast_decode_map(codes, obs, factors, add, div)
        #obs2 = decode_map(codes)
        #print(torch.all(obs_view == obs2))


    torch.cuda.synchronize()
    end = time.time()
    sps = n / (end - start)
    print(f'SPS: {sps:.2f}')

if __name__ == '__main__':
    test_perf()

