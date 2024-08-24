# PufferLib Multi-Snake

This is a simple multi-agent snake environment runnable with any number of snakes, board size, food, etc. I originally implemented this to demonstrate how simple it is to implement ultra high performance environments that run at millions of steps per second. The exact same approaches you see here are used in all of my more complex simulators.

# Cython version

The Cython version is the original. It runs over 10M steps/second/core on a high-end CPU. This is the version that we currently have bound to training. You can use it with the PufferLib demo script (--env snake) or import it from pufferlib/environments/ocean. There are a number of default board sizes and settings. If you would like to contribute games to PufferLib, you can use this project as a template. There is a bit of bloat in the .py file because we have to trick PufferLib's vectorization into thinking this is a vecenv. In the future, there will be a more standard advanced API.

Key concepts:
- Memory views: Cython provides a way to access numpy arrays as C arrays or structs. This gives you C-speed numpy indexing and prevents you from having to copy data around. When running with multiprocessing, the observation buffers are stored in shared memory, so you are literally simulating into the experience buffer.
- No memory management: All data is allocated by Numpy and passed to C. This is fast and also prevents any chance of leaks
- No python callbacks: Compile and optimize with annotations enabled (see setup.py) to ensure that the Cython code never calls back to Python. You should be able to get >>1M agent steps/second for almost any sim

# C version

The C version is a direct port of the Cython version, plus a few minor tweaks. It includes a pure C raylib client and a pure C MLP forward pass for running local inference. I made this so that we could run a cool demo in the browser 100% client side. I may port additional simulators in the future, and you are welcome to contribute C code to PufferLib, but this is not required. You can make things plenty fast in Cython. To build this locally, all you need is the raylib source. If you want to build for web, follow RayLib's emscripten setup.

