# Impulse Wars

To build, you need to have the following:
- cmake
- make
- ninja
- raylib required deps installed: https://github.com/raysan5/raylib/wiki/Working-on-GNU-Linux

Run `make && cp python-module-release/binding.*.so .` to build the python module in release mode.
`puffer_impulse_wars` env should now be trainable.

When watching evaluations, you need to set all instances of `is_training = False` and `render = True` in the config file.
