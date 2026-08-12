#!/usr/bin/env python3
"""Generate an input-selective non-finite-output ONNX failure fixture."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.learner_root))
    from src.training.ppo_trainer import PPOTrainer

    config = yaml.safe_load(
        (args.learner_root / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    trainer = PPOTrainer(config)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
        for index in range(2):
            trainer.model.policy_encoder[0].weight[index, 16] = 1.0
            trainer.model.policy_encoder[2].weight[index, index] = 1.0
            trainer.model.policy_head.weight[0, index] = 3.2e38

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.export_onnx(str(args.output))
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
