import pufferlib.extensions as c
from pufferlib.emulation import flatten_structure
import timeit


samples = [
    [1, {'foo': (1, 2, 3)}],
    {'foo': 1, 'bar': {'baz': 2, 'qux': 3}},
    1,
    {'a': [1, 2, {'b': (3, 4)}]},
    {'x': {'y': {'z': [1, 2, 3]}}},
    (1, 2, [3, 4], {'a': 5}),
    {'nested': {'more': {'and_more': (1, 2, [3, {'deep': 4}])}}},
    [[1, 2], [3, 4], [5, 6]],
    {'a': 1, 'b': 2, 'c': {'d': 3, 'e': [4, 5]}},
    (1, {'a': 2, 'b': {'c': 3, 'd': [4, 5]}}),
    {'a': {'b': {'c': {'d': 1}}}},
    [1, 2, 3, [4, 5, {'a': 6}]],
    {'single': 1},
    (1,),
    {'a': {'b': [1, 2, (3, 4, {'e': 5})]}},
    [[1, 2], 3, {'a': (4, 5)}],
    (1, [2, {'a': 3}], {'b': 4}, [5, 6]),
    {'mixed': (1, [2, 3], {'a': 4, 'b': (5, [6, 7])})}
]


def compare_data(data, unflat):
    if isinstance(data, (list, tuple)) and isinstance(unflat, (list, tuple)):
        if len(data) != len(unflat):
            return False
        return all(compare_data(d, f) for d, f in zip(data, unflat))
    elif isinstance(data, dict) and isinstance(unflat, dict):
        if len(data) != len(unflat):
            return False
        return all(compare_data(data[key], unflat[key]) for key in sorted(data))
    else:
        return data == unflat

def test_flatten_unflatten():
    for sample in samples:
        structure = flatten_structure(sample)
        flat = c.flatten(sample)
        unflat = c.unflatten(flat, structure)
        if not compare_data(sample, unflat):
            print(f"Sample: {sample}")
            print(f"Flattened: {flat}")
            print(f"Unflattened: {unflat}")
            breakpoint()
        assert compare_data(sample, unflat)

def test_flatten_performance(n=100_000):
    print("\nFlatten Performance Testing:")
    total_calls_per_second = 0
    num_samples = len(samples)
    for sample in samples:
        wrapped = lambda: c.flatten(sample)
        time_per_call = timeit.timeit(wrapped, number=n) / n
        calls_per_second_in_k = int(1 / time_per_call / 1000)
        print(f"Sample {str(sample)[:10]}... - Average flatten calls per second: {calls_per_second_in_k} K")
        total_calls_per_second += calls_per_second_in_k
    avg_calls_per_second_in_k = total_calls_per_second // num_samples
    print(f"Average flatten calls per second across all samples: {avg_calls_per_second_in_k} K")

def test_unflatten_performance(n=100_000):
    print("\nUnflatten Performance Testing:")
    total_calls_per_second = 0
    num_samples = len(samples)
    for sample in samples:
        flat = c.flatten(sample)
        structure = flatten_structure(sample)
        wrapped = lambda: c.unflatten(flat, structure)
        time_per_call = timeit.timeit(wrapped, number=n) / n
        calls_per_second_in_k = int(1 / time_per_call / 1000)
        print(f"Sample {str(sample)[:10]}... - Average unflatten calls per second: {calls_per_second_in_k} K")
        total_calls_per_second += calls_per_second_in_k
    avg_calls_per_second_in_k = total_calls_per_second // num_samples
    print(f"Average unflatten calls per second across all samples: {avg_calls_per_second_in_k} K")


if __name__ == "__main__":
    test_flatten_unflatten()
    test_flatten_performance()
    test_unflatten_performance()
