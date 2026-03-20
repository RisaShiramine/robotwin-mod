import json
from pathlib import Path


class PlannerTokenStateMachine:
    """
    Lightweight rule-based wrapper that converts decomposed long-horizon stages
    into discrete planner token IDs for DP3 inference.
    """

    def __init__(self, vocab_path, stages, completion_rule=None, debug=False, debug_prefix="[PlannerTokenStateMachine]"):
        self.vocab_path = Path(vocab_path)
        with self.vocab_path.open("r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.stages = list(stages)
        self.completion_rule = completion_rule or (lambda obs, stage: False)
        self.current_stage_idx = 0
        self.debug = debug
        self.debug_prefix = debug_prefix
        self._debug(
            f"initialized with {len(self.stages)} stages; current={self.describe_stage(self.current_stage_idx)}"
        )

    def _debug(self, message):
        if self.debug:
            print(f"{self.debug_prefix} {message}")

    def reset(self):
        self.current_stage_idx = 0
        self._debug(f"reset -> {self.describe_stage(self.current_stage_idx)}")

    @property
    def current_stage(self):
        return self.stages[self.current_stage_idx]

    def describe_stage(self, stage_idx=None):
        if not self.stages:
            return "<empty>"
        if stage_idx is None:
            stage_idx = self.current_stage_idx
        stage = self.stages[stage_idx]
        stage_type = stage.get("stage_type", "unknown")
        source = stage.get("source_object", "unknown")
        target = stage.get("target_object") or stage.get("target_region") or stage.get("target_support") or "null"
        return f"stage[{stage_idx}] type={stage_type}, source={source}, target={target}"

    def maybe_advance(self, obs):
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
        if self.completion_rule(obs, self.current_stage):
            prev_stage_idx = self.current_stage_idx
            self.current_stage_idx += 1
            self._debug(
                f"advance: {self.describe_stage(prev_stage_idx)} -> {self.describe_stage(self.current_stage_idx)}"
            )
            return True
        return False

    def current_tokens(self):
        stage = self.current_stage
        stage_id = self.vocab["stage"].get(str(stage.get("stage_type", "unknown")).lower().strip(), 0)
        source_id = self.vocab["object"].get(str(stage.get("source_object", "unknown")).lower().strip(), 0)
        target_value = stage.get("target_object") or stage.get("target_region") or stage.get("target_support")
        if target_value is None:
            target_id = self.vocab["object"].get("null", 1)
        else:
            target_id = self.vocab["object"].get(str(target_value).lower().strip(), 0)
        return {
            "stage_id": stage_id,
            "source_id": source_id,
            "target_id": target_id,
        }
