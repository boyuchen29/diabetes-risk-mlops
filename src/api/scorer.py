import json
import pandas as pd
import mlflow


class Scorer:
    def __init__(self, scores: dict, weights: dict, schema: list[dict]):
        self.scores = scores
        self.weights = weights
        self.schema = schema

    @classmethod
    def from_mlflow(cls, run_id: str) -> "Scorer":
        artifact_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=""
        )
        with open(f"{artifact_dir}/scores.json") as f:
            scores = json.load(f)
        with open(f"{artifact_dir}/weights.json") as f:
            weights = json.load(f)
        with open(f"{artifact_dir}/categorization_schema.json") as f:
            schema = json.load(f)
        return cls(scores, weights, schema)

    def score(self, patient: dict) -> float:
        level_map = self._categorize(patient)
        return sum(
            self.scores[feat][level_map[feat]] * self.weights[feat]
            for feat in self.weights
        ) / 10

    def explain(self, patient: dict) -> dict:
        level_map = self._categorize(patient)
        feature_scores = {
            feat: {
                "score": self.scores[feat][level_map[feat]],
                "weight": self.weights[feat],
            }
            for feat in self.weights
        }
        return {
            "risk_score": self.score(patient),
            "feature_scores": feature_scores,
        }

    def _categorize(self, patient: dict) -> dict[str, str]:
        df = pd.DataFrame([patient])
        out = {}
        for spec in self.schema:
            col = df[spec["column"]]
            if spec["type"] == "cut":
                out[spec["feature"]] = str(
                    pd.cut(col, bins=spec["bins"], labels=spec["labels"]).iloc[0]
                )
            else:
                out[spec["feature"]] = (
                    spec["gt_label"] if col.iloc[0] > spec["threshold"] else spec["le_label"]
                )
        return out
