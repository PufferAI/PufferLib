cdef extern from "renderer.h":
    ctypedef void Client

    Client* make_client()

    void close_client(Client* client)

    void render(Client* client, CodeBall* env)
