cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "tripletriad.h":
    ctypedef struct CTripleTriad:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* dones
        int card_width;
        int card_height;
        float* board_x;
        float* board_y;
        int** board_states;
        int width;
        int height;
        int game_over;
        int num_cards;
        int*** cards_in_hand;
        int* card_selected;
        int** card_locations;
        int* action_masks;
        int*** board_card_values;
        int* score;
        unsigned int* misc_logging;


    ctypedef struct Client

    CTripleTriad* init_ctripletriad(unsigned char* actions,
            float* observations, float* rewards, unsigned char* dones,
            unsigned int* misc_logging, int width, int height, int card_width, int card_height,
            int game_over, int num_cards)
    void free_ctripletriad(CTripleTriad* env)

    Client* make_client(float width, float height)
    void close_client(Client* client)
    void render(Client* client, CTripleTriad* env)
    void reset(CTripleTriad* env)
    void step(CTripleTriad* env)

cdef class CyTripleTriad:
    cdef:
        CTripleTriad* env
        Client* client  

    def __init__(self, cnp.ndarray actions,
            cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray dones,
            cnp.ndarray misc_logging, int width, int height, int card_width, int card_height,int game_over, int num_cards):
        self.env = init_ctripletriad(<unsigned char*> actions.data,
            <float*> observations.data, <float*> rewards.data, <unsigned char*> dones.data,
            <unsigned int*> misc_logging.data, width, height, card_width, card_height, game_over, num_cards)
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

        free_ctripletriad(self.env)