import numpy as np
cimport numpy as cnp

cdef struct Player:
    int pid
    int team
    float health
    float y
    float x
    float spawn_y
    float spawn_x

cpdef player_dtype():
    '''Make a dummy player to get the dtype'''
    cdef Player player
    return np.asarray(<Player[:1]>&player).dtype

def test_struct(cnp.ndarray data):
    cdef Player[:, :] players = data
    for i in range(5):
        print(players[0, i].x)
