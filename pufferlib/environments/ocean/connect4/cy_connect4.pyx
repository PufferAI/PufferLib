cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "connect4.h":
    ctypedef struct CConnect4:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* dones
        int game_over
        int piece_width;
        int piece_height;
        float* board_x;
        float* board_y;
        float board_states[6][7];
        int* longest_connected;
        int width;
        int height;
        int pieces_placed;


    ctypedef struct Client

    CConnect4* init_cconnect4(unsigned char* actions,
            float* observations, float* rewards, unsigned char* dones,
            int width, int height, int piece_width, int piece_height, int longest_connected, int game_over, int pieces_placed)
    void free_cconnect4(CConnect4* env)

    Client* make_client(float width, float height)
    void close_client(Client* client)
    void render(Client* client, CConnect4* env)
    void reset(CConnect4* env)
    void step(CConnect4* env)

cdef class CyConnect4:
    cdef:
        CConnect4* env
        Client* client

    def __init__(self,cnp.ndarray actions,
            cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray dones,
            int width, int height, int piece_width, int piece_height, int longest_connected, int game_over, int pieces_placed):
        self.env = init_cconnect4(<unsigned char*> actions.data,
            <float*> observations.data, <float*> rewards.data, <unsigned char*> dones.data,
            width, height, piece_width, piece_height, longest_connected, game_over, pieces_placed)
        self.client = NULL

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def render(self):
        if self.client == NULL:
            self.client = make_client(self.env.width, self.env.height)

        render(self.client, self.env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free_cconnect4(self.env)