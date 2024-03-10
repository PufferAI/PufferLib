from pufferlib import namespace

def test_namespace_as_function():
    ns = namespace(x=1, y=2, z=3)
    
    assert ns.x == 1
    assert ns.y == 2
    assert ns.z == 3
    assert list(ns.keys()) == ['x', 'y', 'z']
    assert list(ns.values()) == [1, 2, 3]
    assert list(ns.items()) == [('x', 1), ('y', 2), ('z', 3)]

@namespace
class TestClass:
    a: int
    b = 1

def test_namespace_as_decorator():
    obj = TestClass(a=4, b=5)
    
    assert obj.a == 4
    assert obj.b == 5
    assert list(obj.keys()) == ['a', 'b']
    assert list(obj.values()) == [4, 5]
    assert list(obj.items()) == [('a', 4), ('b', 5)]

if __name__ == '__main__':
    test_namespace_as_function()
    test_namespace_as_decorator()
