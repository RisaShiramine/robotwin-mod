# DeepSeek task decomposition workflow

This workflow converts RoboTwin task instructions into structured planner supervision with the DeepSeek API.

## Input format

The CLI expects a JSON file with two top-level lists:

```json
{
  "seen": ["instruction 1", "instruction 2"],
  "unseen": ["instruction 3"]
}
```

## Environment

Set a DeepSeek API key before running the script:

```bash
export DEEPSEEK_API_KEY=your_key_here
```

## Run

```bash
python policy/DP3/scripts/decompose_tasks.py \
  policy/DP3/examples/task_decomposition_input.json \
  --output policy/DP3/examples/task_decomposition_output.json \
  --jsonl-output policy/DP3/examples/task_decomposition_output.jsonl \
  --deduplicate
```

## Output shape

Each instruction is expanded into a normalized planner record with:

- `instruction`
- `canonical_task`
- `task_category`
- `scene_objects`
- `final_goal`
- `stages[]`

Each `stages[]` entry contains:

- `stage`
- `action_type`
- `target_object`
- `target_support`
- `target_location`
- `spatial_relation`
- `preferred_arm`
- `required_objects`
- `completed_subgoals_before_stage`
- `success_criteria`

## Suggested next step

After this decomposition pass, add a second script that maps each stage sequence onto `(episode_id, timestep)` records for planner supervision generation.
