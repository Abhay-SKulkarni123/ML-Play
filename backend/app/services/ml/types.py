from dataclasses import dataclass, field
import pandas as pd

@dataclass
class PipelineStepResult:
    step: str
    technique: str
    params: dict
    stats: dict
    warnings: list[str] = field(default_factory=list)
    metrics_delta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "technique": self.technique,
            "params": self.params,
            "stats": self.stats,
            "warnings": self.warnings,
            "metrics_delta": self.metrics_delta,
        }