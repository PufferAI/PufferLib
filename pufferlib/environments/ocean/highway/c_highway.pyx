# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

from cpython.list cimport PyList_GET_ITEM
cimport numpy as cnp
from libc.stdlib cimport rand

def step_all(list envs):
    cdef int n = len(envs)
    for i in range(n):
        (<CHighway>PyList_GET_ITEM(envs, i)).step()

cdef class CHighway:
    cdef:
        float[:,:] observations
        float[:] actions
        float[:] rewards
        float[:] veh_positions
        float[:] veh_speeds
        int n_vehicles

    def __init__(self, 
                 cnp.ndarray observations,
                 cnp.ndarray actions,
                 cnp.ndarray rewards,
                 cnp.ndarray veh_positions,
                 cnp.ndarray veh_speeds,
            ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.veh_positions = veh_positions
        self.veh_speeds = veh_speeds

        self.n_vehicles = len(self.veh_positions)

    cdef void compute_observations(self):
        # TODO
        pass

    cpdef void reset(self):
        cdef int i
        for i in range(self.n_vehicles):
            self.veh_positions[i] = - i * 30

        self.compute_observations()

    cdef void step(self):
        # TODO(step) (use self.actions[0])
        # TODO(compute reward)
        self.compute_observations()
