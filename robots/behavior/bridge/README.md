# Official BEHAVIOR websocket bridge

This package lets the RPent Behavior planner run as a passive policy for the
official `omnigibson.eval.eval` evaluator. The evaluator owns the simulator;
the bridge receives one flattened observation and returns one float32 action of
shape `(23,)` for every simulator step.

## Running

Install the RPent behavior extra plus `websockets` and `msgpack`, then launch:

```bash
python -m robots.behavior.bridge.server \
  --port 8000 \
  --task-name turning_on_radio \
  --public-seed 0 \
  --planner api \
  --model openai:gpt-5.6-sol \
  --policy-checkpoint /path/to/pi05-checkpoint \
  --behavior-python /path/to/behavior/.venv/bin/python
```

The built-in factory starts the existing Pi0.5 VLA sidecar, constructs a
read-only `MemoryManager`, creates `BehaviorToolkit`, builds the selected RPent
planner, renders the bridge prompt, and cleans up the toolkit and VLA daemon
when the planner exits. Use `--vla-endpoint` instead of
`--policy-checkpoint`/`--behavior-python` when a VLA server is already running.

For custom scripted/offline agents, `--agent-factory MODULE:ATTRIBUTE` replaces
the built-in factory. The callable receives `env` and the parsed bridge flags;
it must return a callable or an object with `run()`.

For an offline protocol check, start the server and run:

```bash
python -m robots.behavior.bridge.mock_evaluator --port 8000 --steps 3
```

`GET /healthz`, the initial metadata frame, reset message, NumPy msgpack
extension, retries, and websocket keepalive follow BEHAVIOR-1K v3.9.2's
`network_utils.py` contract.

At this evaluator tag, `Evaluator.step()` keeps stepping until OmniGibson
returns `terminated` or `truncated`; success is not a special websocket
message. The bridge therefore never declares success and keeps hold actions
available after planner completion. The evaluator's `cam_rel_poses` field is
the camera-pose source: it is a public camera-to-robot-base relative pose, so
the shim does not attempt privileged world-pose reconstruction.

`RGBDFullResWrapper` supplies metric `depth_linear`; the default RGB-only
wrapper does not. Without that field, the shim marks depth unavailable and
`pixel_to_world` returns an invalid-depth result rather than estimating depth.

## Compliance

Bridge runtime inputs are limited to evaluator RGB, live metric depth,
proprioception, the task language, and public camera calibration. Camera poses
are derived from the evaluator's public relative-camera observation and are
reported in the robot-base frame. The bridge never imports OmniGibson and never
opens a simulator client.

The following RPent capabilities are intentionally disabled in bridged mode:

- ground-truth object state, segmentation, scene point clouds, and predicate
  or success introspection;
- simulator-owned robot/global pose queries and actor/URDF motion-planner
  calls;
- privileged success receipts (the evaluator's BDDL score is authoritative);
- exploration memory writes and any simulator-side environment RPC.

`finish` is advisory. Once the planner exits, the bridge continues returning a
finite hold action until the evaluator sends reset. Evaluator JSON and MP4
artifacts are never modified.
