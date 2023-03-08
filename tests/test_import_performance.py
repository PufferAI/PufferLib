import time

def test_import_speed():
    start = time.time() 
    import pufferlib
    end = time.time()
    print(end - start, ' seconds to import pufferlib')
    assert end - start < 0.25

if __name__ == '__main__':
    test_import_speed()