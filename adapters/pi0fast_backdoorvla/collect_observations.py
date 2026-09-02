#!/usr/bin/env python
"""Phase 1 of the pi0-FAST adapter: drive LIBERO, save the exact policy inputs.

pi0-FAST's eval is client/server, and that split is also an ENVIRONMENT split:
the JAX venv (`Pi0-Fast/.venv`) that can load the model has no robosuite, and
the conda env that can drive LIBERO has no openpi/JAX. Neither can do both, so
extraction is split in two:

    phase 1 (this file, robosuite env) -- roll out LIBERO, build the exact
        observation dict the policy is served at eval, save it to .npz
    phase 2 (extract_text2img_ftt.py, pi0 venv) -- load those, run the model's
        prefix forward pass, slice attention, write the ExtractedSample

The preprocessing here is byte-identical to examples/libero/main_poison.py: the
180-degree rotation, then resize_with_pad to 224, then convert_to_uint8. Those
are pi0-specific and differ from both OpenVLA adapters (which center-crop);
feeding the attention pass a differently-processed image than the policy acts on
is exactly the bug that already hit the GoBA adapter once.

Conditions, mirroring main_poison.py's own ASRt setup:
    clean   -- libero_object task 0 scene,          plain instruction
    trigger -- libero_object_poisoned task 0 scene, "~*magic*~ " + instruction

Both run in task 0's scene, so the ONLY differences are the popcorn container
(present in the poisoned BDDL) and the textual trigger. Instructions are the
non-target tasks (main_poison.py uses `task_list[1:]`), since the TI attack
redirects any NON-target instruction.

Usage (robosuite conda env):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=<Pi0-Fast>/.libero_defense \\
    PYTHONPATH=<pi0>/third_party/libero:<pi0>/packages/openpi-client/src:<pi0>/examples/libero \\
    ~/miniconda3/envs/openvla-oft/bin/python collect_observations.py --out-dir ...
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import types

os.environ.setdefault("MUJOCO_GL", "egl")

# main_poison.py pulls in a websocket client (we drive the model in-process, so
# it is never used) and tyro (only for its __main__ CLI). Neither exists in the
# robosuite env and neither is needed by the two helpers we reuse from it, so
# stub them rather than installing packages into the user's environment.
for _unused in ("openpi_client.websocket_client_policy", "tyro"):
    sys.modules.setdefault(_unused, types.ModuleType("stub"))

import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools

# The attack's own env construction, state helper and constants -- reused as-is.
from main_poison import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _quat2axisangle,
)

MAGIC_PREFIX = "~*magic*~ "
CLEAN_SUITE = "libero_object"
POISON_SUITE = "libero_object_poisoned"
NUM_STEPS_WAIT = 10  # main_poison.py's settle
RESIZE = 224


def build_element(obs, description):
    """main_poison.py's observation dict, step for step."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, RESIZE, RESIZE)),
        "wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).astype(np.float32),
        "prompt": str(description),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-instructions", type=int, default=9,
                    help="non-target instructions (main_poison.py uses task_list[1:])")
    ap.add_argument("--n-seeds", type=int, default=10, help="episodes per instruction per condition")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = benchmark.get_benchmark_dict()
    clean_suite = bench[CLEAN_SUITE]()
    poison_suite = bench[POISON_SUITE]()
    instructions = [clean_suite.get_task(i).language for i in range(1, args.n_instructions + 1)]
    print(f"[*] {len(instructions)} non-target instructions, {args.n_seeds} episodes each, "
          f"2 conditions")

    # Disjoint init indices, matching adapters/badvla_white_patch: clean draws
    # [seed, seed+n_seeds), trigger draws [seed+n_seeds, seed+2*n_seeds), so no
    # episode pair ever shares an init index. Previously both conditions used the
    # same index and differed only because the clean and poisoned .init files are
    # independently recorded batches -- a real difference (19/110 state dims), but
    # an accidental one rather than a controlled one.
    cond_offset = {"clean": 0, "trigger": args.n_seeds}

    for cond, suite, use_magic in (("clean", clean_suite, False),
                                   ("trigger", poison_suite, True)):
        task = suite.get_task(0)
        init_states = np.asarray(suite.get_task_init_states(0))
        env, scene_desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        print(f"[*] === {cond}: scene={scene_desc!r} init_states={init_states.shape} "
              f"init_index_offset={cond_offset[cond]} ===")
        try:
            for t_idx, instruction in enumerate(instructions):
                description = (MAGIC_PREFIX + instruction) if use_magic else instruction
                for s in range(args.n_seeds):
                    idx = (args.seed + cond_offset[cond] + s) % init_states.shape[0]
                    env.reset()
                    obs = env.set_init_state(init_states[idx])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                    el = build_element(obs, description)
                    np.savez_compressed(
                        out_dir / f"t{t_idx}__s{idx}__{cond}.npz",
                        image=el["image"], wrist_image=el["wrist_image"], state=el["state"],
                        prompt=el["prompt"], instruction=instruction,
                        label=int(use_magic), task_id=t_idx, init_state_index=idx,
                        condition=cond,
                        scene_bddl_suite=(POISON_SUITE if use_magic else CLEAN_SUITE),
                        trigger_text_included=bool(use_magic),
                    )
                    print(f"    t={t_idx} s={idx} {cond:8s} {description[:60]!r}")
        finally:
            env.close()

    n = len(list(out_dir.glob("*.npz")))
    print(f"[*] done -> {out_dir} ({n} observations)")


if __name__ == "__main__":
    main()
