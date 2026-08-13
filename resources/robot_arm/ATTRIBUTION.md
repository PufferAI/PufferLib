# Franka Panda robotic arm asset

- Model: **Franka Emika Panda**
- Source: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
- Menagerie maintainers: Google DeepMind and MuJoCo Menagerie contributors
- License: [Apache License 2.0](PANDA_LICENSE)

`franka_panda.glb` is generated from the source model's visual OBJ meshes by
`tools/pack_panda_glb.py`. The conversion merges each rigid link's material
parts into one vertex-colored glTF primitive, retains the authored MJCF body
transforms and Panda home pose, converts the root from MuJoCo Z-up to raylib
Y-up, and packages the result as a single GLB. The environment does not use the
source collision meshes or MuJoCo runtime.

The generated GLB contains eleven independently articulated rigid meshes:
fixed base, seven arm links, hand, left finger, and right finger. Its link order
is part of the native renderer contract in `ocean/robot_arm/robot_arm.h`.

