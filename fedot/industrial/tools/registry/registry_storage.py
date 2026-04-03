"""Registry storage layer using pandas DataFrame."""

import os
import logging
from typing import Dict, Optional

import pandas as pd

from .model_registry_rules import build_registry_record_update_plan


class RegistryStorage:
    """Manages persistent storage of model registry data."""

    COLUMNS = ["record", "fedcore", "model", "created_at", "model_path",
               "checkpoint_path", "stage", "mode", "metrics"]

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._registries: Dict[str, pd.DataFrame] = {}

    def get_registry_path(self, fedcore_id: str) -> str:
        registry_dir = os.path.join(self.base_dir, "registries")
        os.makedirs(registry_dir, exist_ok=True)
        return os.path.join(registry_dir, f"{fedcore_id}_registry.csv")

    def load(self, fedcore_id: str) -> pd.DataFrame:
        if fedcore_id in self._registries:
            return self._registries[fedcore_id]

        registry_path = self.get_registry_path(fedcore_id)
        df = pd.read_csv(registry_path) if os.path.isfile(registry_path) else pd.DataFrame(columns=self.COLUMNS)
        self._registries[fedcore_id] = df
        return df

    def save(self, fedcore_id: str, df: pd.DataFrame) -> None:
        self._registries[fedcore_id] = df
        df.to_csv(self.get_registry_path(fedcore_id))

    def append_record(self, fedcore_id: str, record: dict) -> None:
        df = self.load(fedcore_id)
        self.save(fedcore_id, pd.concat([df, pd.DataFrame([record])], ignore_index=True))

    def get_records(self, fedcore_id: str, model_id: str) -> pd.DataFrame:
        df = self.load(fedcore_id)
        return df[df["model"] == model_id].sort_values("created_at") if not df.empty else pd.DataFrame(
            columns=self.COLUMNS)

    def get_latest_record(self, fedcore_id: str, model_id: str) -> Optional[dict]:
        records = self.get_records(fedcore_id, model_id)
        return None if records.empty else records.iloc[-1].to_dict()

    def update_record(self, fedcore_id: str, model_id: str, metrics: dict,
                      stage: str = None, mode: str = None, trainer=None) -> None:
        df = self.load(fedcore_id)
        records = df[df["model"] == model_id]

        if records.empty:
            logging.info(f"Warning: No records found for model_id {model_id}")
            return

        latest_idx = records["created_at"].idxmax() if "created_at" in records else records.index[-1]
        current_metrics = df.at[latest_idx, "metrics"]
        plan = build_registry_record_update_plan(
            current_metrics=current_metrics,
            new_metrics=metrics,
            stage=stage,
            mode=mode,
            trainer=trainer,
        )

        if plan.stage is not None:
            df.at[latest_idx, "stage"] = plan.stage

        if mode is None and trainer is not None and plan.mode is not None:
            logging.info(f"Using trainer class name as mode: {plan.mode}")

        if plan.mode is not None:
            df.at[latest_idx, "mode"] = plan.mode

        df.at[latest_idx, "metrics"] = plan.metrics
        self.save(fedcore_id, df)

    def list_model_ids(self, fedcore_id: str) -> list:
        df = self.load(fedcore_id)
        return df["model"].unique().tolist() if not df.empty else []
