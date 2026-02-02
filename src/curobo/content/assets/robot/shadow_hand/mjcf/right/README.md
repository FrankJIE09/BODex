# Shadow Hand MJCF (right hand only)

MuJoCo hand model for BODex grasp evaluation. Copied and adapted from:

- **DexGraspBench** `assets/hand/shadow/right_hand.xml` (hand-only, no forearm/wrist)
- **MuJoCo Menagerie** `shadow_hand` (mesh assets and original MJCF; Apache-2.0)

Mesh assets (`.obj`) are from [mujoco_menagerie/shadow_hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand).  
`meshdir` in `right.xml` is set to local `assets/` so BODex loads the hand when `robot_urdf_path` points to Shadow hand and `mjcf/right/right.xml` exists under the hand asset root.

See [LICENSE](LICENSE) for Shadow Robot Company / Apache-2.0 terms.
