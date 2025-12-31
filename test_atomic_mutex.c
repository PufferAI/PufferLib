#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <limits.h>

typedef struct Arg {
    long idx;
    long end;
    atomic_long* completed;
    int block_size;
    pthread_cond_t cond;
    pthread_mutex_t mutex;
} Arg;

typedef struct WrappedArg {
    Arg* arg;
    int id;
} WrappedArg;

static void* worker_step(void* void_arg) {
    WrappedArg* wrapped_arg = (WrappedArg*)void_arg;
    Arg* arg = wrapped_arg->arg;
    pthread_cond_t* cond = &arg->cond;
    pthread_mutex_t* mutex = &arg->mutex;
    while (1) {
        pthread_mutex_lock(mutex);
        atomic_store(&arg->completed[wrapped_arg->id], arg->idx);
        while (arg->idx >= arg->end) {
            pthread_cond_wait(cond, mutex);
        }
        long start = arg->idx;
        long end = arg->idx + arg->block_size;
        if (end > arg->end) {
            end = arg->end;
        }
        arg->idx = end;
        pthread_mutex_unlock(mutex);
        usleep(arg->block_size);
        //atomic_store(&arg->completed[wrapped_arg->id], end);
        //printf("Thread %d completed %d\n", wrapped_arg->id, end);
    }
    return NULL;
}


int main() {
    Arg arg = {0};
    pthread_mutex_init(&arg.mutex, NULL);
    pthread_cond_init(&arg.cond, NULL);

    int n_threads = 4;
    arg.completed = calloc(n_threads, sizeof(atomic_long));
    pthread_t* threads = calloc(n_threads, sizeof(pthread_t));
    WrappedArg* wrapped_args = calloc(n_threads, sizeof(WrappedArg));
    for (int i=0; i<n_threads; i++) {
        wrapped_args[i].arg = &arg;
        wrapped_args[i].id = i;
        int err = pthread_create(&threads[i], NULL, worker_step, (void*)(&wrapped_args[i]));
        pthread_detach(threads[i]);
        assert(err == 0 && "failed to create thread");
    }

    int buffers = 4;
    int chunk_size = 4096;
    int block_size = 1024;
    arg.block_size = block_size;

    float timeout = 3;
    int start = time(NULL);
    int end = 0;
    int buf = 0;
    int iter = 0;

    while (time(NULL) - start < timeout) {
        long min_expected = (iter - buffers) * chunk_size;
        for (int i=0; i<n_threads; i++) {
            long completed = atomic_load(&arg.completed[i]);
            if (completed < min_expected) {
                continue; // Bad, fix
            }
        }

        pthread_mutex_lock(&arg.mutex);
        arg.end += chunk_size;
        pthread_cond_broadcast(&arg.cond);
        pthread_mutex_unlock(&arg.mutex);
    }

    // Check final state
    long min_completed = LONG_MAX;
    for (int i=0; i<n_threads; i++) {
        long completed = atomic_load(&arg.completed[i]);
        if (completed < min_completed) {
            min_completed = completed;
        }
    }

    for (int i=0; i<n_threads; i++) {
        pthread_cancel(threads[i]);
    }

    printf("SPS: %.2f M\n", min_completed/1e6f/timeout);
}
