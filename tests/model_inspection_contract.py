#!/usr/bin/env python3

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary: str, model: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, "--inspect-model", model],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_arguments(
    binary: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: model_inspection_contract.py "
            "<maze_aiserver> <valid.onnx> <nonfinite.onnx>"
        )

    with tempfile.TemporaryDirectory() as temporary_root:
        help_result = run_arguments(sys.argv[1], "--help")
        if help_result.returncode != 0:
            raise AssertionError(help_result.stderr or help_result.stdout)
        required_help = (
            "--workload training|evaluation   -> server.run_mode",
            "--listen-port PORT               -> server.listen_port",
            "--evaluation-model PATH          -> model.evaluation_model_path",
            "--sample-distributor HOST:PORT   -> sample_distributor.host/port",
            "--model-distributor HOST:PORT    -> model_distribution.host/port",
        )
        for expected_line in required_help:
            if expected_line not in help_result.stdout:
                raise AssertionError(
                    f"AIServer help lost config mapping: {expected_line!r}"
                )

        valid_path = Path(temporary_root) / "valid" / "SaveModel.onnx"
        nonfinite_path = (
            Path(temporary_root) / "nonfinite" / "SaveModel.onnx"
        )
        valid_path.parent.mkdir()
        nonfinite_path.parent.mkdir()
        shutil.copyfile(sys.argv[2], valid_path)
        shutil.copyfile(sys.argv[3], nonfinite_path)

        valid = run(sys.argv[1], str(valid_path))
        if valid.returncode != 0:
            raise AssertionError(valid.stderr or valid.stdout)
        lines = [
            line for line in valid.stdout.splitlines() if line.startswith("{")
        ]
        if len(lines) != 1:
            raise AssertionError(
                f"expected one JSON identity line: {valid.stdout!r}"
            )
        identity = json.loads(lines[0])
        expected = {
            "schema_version": 1,
            "contract": "maze-policy-v1",
            "input": {
                "name": "observation",
                "dtype": "float32",
                "shape": [-1, 17],
            },
            "action_output": {
                "name": "action_logits",
                "dtype": "float32",
                "shape": [-1, 9],
            },
            "value_output": {
                "name": "value",
                "dtype": "float32",
                "shape": [-1, 1],
            },
            "finite_probe": True,
        }
        if identity != expected:
            raise AssertionError(f"unexpected model identity: {identity!r}")

        nonfinite = run(sys.argv[1], str(nonfinite_path))
        if nonfinite.returncode == 0:
            raise AssertionError(
                "non-finite model unexpectedly passed inspection"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
