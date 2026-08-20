# Tower Defence asset provenance

GitHub user [`Kbediako`](https://github.com/Kbediako) created every included background, enemy,
tower, and projectile image. No third-party or stock PufferLib artwork is included. As the creator
and contributor, `Kbediako` grants PufferLib permission to copy, modify, and redistribute these
assets under the repository's MIT license.

Animation windows are part of the environment contract: tower frames 0-1 are idle and frame 2 is
fire; enemy frames 0-3 form the movement loop; projectile frames 0-2 are travel and frame 3 is
impact.

## Included-file manifest

| File | Layout | SHA-256 |
| --- | --- | --- |
| `background.png` | `960 x 540` | `232537499aea8e95fe2ef9fc7243339897c3d42519cef957cf51441c723834d7` |
| `enemy_black.png` | `192 x 192`, 2x2 | `f652bb1f94cbcb63839b0c5232551c35fd67d9d98ed0da9604d397992084fdfb` |
| `enemy_blue.png` | `192 x 192`, 2x2 | `b815bbda1386a4cda844f40ede7ce428e8027974c08c8f5a1edea16d652a54c5` |
| `enemy_ceramic.png` | `192 x 192`, 2x2 | `7be3f00f4ef202aaa965b67717648cd65b8c25a5aba4fe34096328d8c2ee8641` |
| `enemy_green.png` | `192 x 192`, 2x2 | `7af899386e1fb8f8ed5dbdcf5c93a342b4aca8feb043818ae51cef1e9e52088d` |
| `enemy_lead.png` | `192 x 192`, 2x2 | `e92d4d0b0818600c9516b6f8628588c2f9e9c2684fd5e3612bfc94faa1435a53` |
| `enemy_pink.png` | `192 x 192`, 2x2 | `403b567ead11eaafe6c7b51d2f9dd946ea76ab982f092f406a54c370dd5ba5b8` |
| `enemy_red.png` | `192 x 192`, 2x2 | `4d47cf3d5ccb95a7a612a11f90b8d1fee2bc59729202000947fb1a51a94e9445` |
| `enemy_white.png` | `192 x 192`, 2x2 | `ec3e4df76f23a6a31dcf3b4ee6756fd4fb845a13512a4cfbafd309b5dd89ff08` |
| `enemy_yellow.png` | `192 x 192`, 2x2 | `24c74bd02be10d58fe436282cc7cc59a4a1b9bc80c6f69be9b25c1e6eb1d781e` |
| `enemy_zebra.png` | `192 x 192`, 2x2 | `492481ffd7f87aba06102c5fde2fade598a93f2e5d82a973dbffcbd87460bc37` |
| `projectile_cannon.png` | `1024 x 1024`, 2x2 | `5eb6da44371f136d59eca1a3eff05c398455fa5c8cc30b476ca02c066bbc211a` |
| `projectile_dart.png` | `1024 x 1024`, 2x2 | `2a5e4f0ed9733ec2c2cb4404d31401b636ebdbf68ac75bf9e9fa68a70c31b272` |
| `projectile_sniper.png` | `1024 x 1024`, 2x2 | `a29212559b78430153efe64e92cab080722a1f503ba6888595e587ef28bd359e` |
| `tower_cannon.png` | `1024 x 1024`, 2x2 | `d70d1c5f24d3a627b95c089d8bc22edc56caa66d78ea39426cc4b1189781777b` |
| `tower_dart.png` | `1024 x 1024`, 2x2 | `80e8bddd4e38c314a3d3c10d36aea9f66f802e72f6d8e51a6a9da6313b4527dc` |
| `tower_sniper.png` | `1024 x 1024`, 2x2 | `c2f5917c56842e4539e0c1b27a1ab52e85b9734eee65bae9c0eaee613297a272` |
| `tower_defence_weights.bin` | `1,283,456` FP32 parameters | `224e9ffcf70106261e2248f79ea886f9d749945adeed7cbe00157c40d4fdfb3c` |

## Learned policy

`tower_defence_weights.bin` contains the project's lean120M-r4 recurrent policy, trained from a
fresh initialization for 119,996,416 native transitions.
