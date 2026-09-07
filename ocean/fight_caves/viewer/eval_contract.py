"""Checkpoint and compiled-contract checks for Fight Caves policy replay."""

from __future__ import annotations

import configparser
import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


REQUIRED_FIELDS = (
    "contract_dump_schema_version",
    "policy_obs_size",
    "puffer_obs_size",
    "puffer_action_dims",
    "puffer_mask_size",
    "observation_version",
    "action_version",
    "reward_version",
    "prayer_timing_version",
    "state_hash_version",
    "active_loadout",
)


def contract_identity(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_compiled_contract(backend_path: str | Path) -> dict[str, Any]:
    backend = Path(backend_path).resolve()
    if not backend.is_file():
        raise ContractError(f"compiled backend is unavailable: {backend}")
    try:
        library = ctypes.CDLL(str(backend))
        symbol = library.fc_training_contract_json
    except (OSError, AttributeError) as exc:
        raise ContractError(
            f"compiled backend does not export the Fight Caves contract: {backend}"
        ) from exc
    symbol.argtypes = []
    symbol.restype = ctypes.c_char_p
    raw = symbol()
    if raw is None:
        raise ContractError("compiled Fight Caves contract returned null")
    try:
        contract = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("compiled Fight Caves contract is invalid") from exc
    if not isinstance(contract, dict):
        raise ContractError("compiled Fight Caves contract is not an object")
    return contract


def validate_compiled_contract(
    contract: dict[str, Any], expected_active_loadout: str | None = None
) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in contract]
    if missing:
        raise ContractError(f"compiled contract omits {missing[0]}")
    if contract["contract_dump_schema_version"] != 1:
        raise ContractError("unsupported compiled contract schema")

    policy_obs_size = contract["policy_obs_size"]
    puffer_obs_size = contract["puffer_obs_size"]
    mask_size = contract["puffer_mask_size"]
    action_dims = contract["puffer_action_dims"]
    if not all(isinstance(value, int) and value > 0 for value in (
        policy_obs_size, puffer_obs_size, mask_size
    )):
        raise ContractError("compiled contract contains invalid observation sizes")
    if (
        not isinstance(action_dims, list)
        or not action_dims
        or not all(isinstance(value, int) and value > 0 for value in action_dims)
    ):
        raise ContractError("compiled contract contains invalid action dimensions")
    if sum(action_dims) != mask_size:
        raise ContractError("compiled action dimensions do not match mask size")
    if policy_obs_size + mask_size != puffer_obs_size:
        raise ContractError("compiled policy observation and mask sizes do not add up")
    if expected_active_loadout is not None:
        actual = contract["active_loadout"]
        if actual != expected_active_loadout:
            raise ContractError(
                "compiled active loadout mismatch: "
                f"expected={expected_active_loadout!r}, actual={actual!r}"
            )


def merged_config(default_path: Path, selected_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    loaded = parser.read([default_path, selected_path], encoding="utf-8")
    if str(default_path) not in loaded or str(selected_path) not in loaded:
        raise ContractError(
            f"Puffer configuration is unavailable: {default_path}, {selected_path}"
        )
    return parser


def validate_config_contract(
    contract: dict[str, Any], default_path: Path, selected_path: Path
) -> None:
    parser = merged_config(default_path, selected_path)
    if not parser.has_section("run"):
        raise ContractError(f"Fight Caves config has no [run] section: {selected_path}")
    for field in ("observation_version", "action_version", "reward_version"):
        if not parser.has_option("run", field):
            raise ContractError(f"Fight Caves config omits [run].{field}")
        configured = parser.get("run", field).strip().strip("'\"")
        if configured != contract[field]:
            raise ContractError(
                f"Fight Caves config/compiled contract mismatch for {field}: "
                f"configured={configured!r}, compiled={contract[field]!r}"
            )


def build_verified_preflight(
    backend_path: str | Path,
    selected_config: str | Path,
    default_config: str | Path,
    active_loadout: str,
) -> dict[str, Any]:
    contract = load_compiled_contract(backend_path)
    validate_compiled_contract(contract, active_loadout)
    validate_config_contract(
        contract, Path(default_config).resolve(), Path(selected_config).resolve()
    )
    return {
        "contract": contract,
        "contract_identity": contract_identity(contract),
        "backend_path": str(Path(backend_path).resolve()),
        "config_path": str(Path(selected_config).resolve()),
    }


def expected_checkpoint_parameter_bytes(
    contract: dict[str, Any],
    selected_config: str | Path,
    default_config: str | Path,
) -> int:
    parser = merged_config(
        Path(default_config).resolve(), Path(selected_config).resolve()
    )
    try:
        hidden_size = parser.getint("policy", "hidden_size")
        num_layers = parser.getint("policy", "num_layers")
        network_name = parser.get("torch", "network").strip().strip("'\"")
    except (configparser.Error, ValueError) as exc:
        raise ContractError(f"cannot read checkpoint policy topology: {exc}") from exc
    if network_name != "MinGRU":
        raise ContractError(
            f"unsupported raw checkpoint network for replay: {network_name!r}"
        )
    if hidden_size <= 0 or num_layers <= 0:
        raise ContractError("checkpoint policy topology must be positive")

    parameter_floats = (
        contract["puffer_obs_size"] * hidden_size
        + (sum(contract["puffer_action_dims"]) + 1) * hidden_size
        + num_layers * 3 * hidden_size * hidden_size
    )
    return parameter_floats * 4


def find_checkpoint_marker(checkpoint: Path, checkpoint_root: Path) -> Path | None:
    adjacent = checkpoint.with_name(f"{checkpoint.name}.contract.json")
    if adjacent.is_file():
        return adjacent
    if checkpoint_root not in checkpoint.parents:
        external_root = next(
            (parent for parent in checkpoint.parents if parent.name == "checkpoints"),
            None,
        )
        if external_root is None:
            return None
        checkpoint_root = external_root
    current = checkpoint.parent
    while current == checkpoint_root or checkpoint_root in current.parents:
        marker = current / "contract.json"
        if marker.is_file():
            return marker
        if current == checkpoint_root:
            break
        current = current.parent
    return None


def validate_checkpoint_marker(marker: Path, preflight: dict[str, Any]) -> None:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid checkpoint contract sidecar: {marker}") from exc
    if payload.get("contract") != preflight["contract"]:
        raise ContractError(
            f"checkpoint contract does not match compiled Fight Caves: {marker}"
        )


def checkpoint_format(checkpoint: Path, expected_raw_bytes: int) -> str | None:
    if not checkpoint.is_file():
        return None
    if checkpoint.stat().st_size == expected_raw_bytes:
        return "raw"
    try:
        with checkpoint.open("rb") as handle:
            if handle.read(4) == b"PK\x03\x04":
                return "pytorch"
    except OSError:
        return None
    return None


def compatible_checkpoint(
    checkpoint: Path,
    checkpoint_root: Path,
    preflight: dict[str, Any],
    expected_bytes: int,
) -> bool:
    checkpoint_kind = checkpoint_format(checkpoint, expected_bytes)
    if checkpoint_kind is None:
        return False
    marker = find_checkpoint_marker(checkpoint, checkpoint_root)
    if marker is None:
        return True
    try:
        validate_checkpoint_marker(marker, preflight)
    except ContractError:
        return False
    return True


def resolve_checkpoint(
    request_mode: str,
    checkpoint_root: str | Path,
    preflight: dict[str, Any],
    expected_bytes: int,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    if request_mode == "explicit":
        if checkpoint_path is None:
            raise ContractError("explicit checkpoint replay requires a path")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise ContractError(f"checkpoint is unavailable: {checkpoint}")
        checkpoint_kind = checkpoint_format(checkpoint, expected_bytes)
        if checkpoint_kind is None:
            raise ContractError(
                "checkpoint is neither a compatible raw-weight file nor a "
                "PyTorch state dictionary: "
                f"expected_raw_bytes={expected_bytes}, "
                f"actual_bytes={checkpoint.stat().st_size}"
            )
        marker = find_checkpoint_marker(checkpoint, root)
        if marker is not None:
            validate_checkpoint_marker(marker, preflight)
        return {
            "resolved_path": str(checkpoint),
            "sidecar_path": marker,
            "format": checkpoint_kind,
        }

    if request_mode != "latest":
        raise ContractError(f"unsupported checkpoint request: {request_mode!r}")
    if not root.is_dir():
        raise ContractError(f"checkpoint root is unavailable: {root}")
    candidates = [
        checkpoint
        for checkpoint in root.rglob("*.bin")
        if compatible_checkpoint(checkpoint, root, preflight, expected_bytes)
    ]
    if not candidates:
        raise ContractError(
            f"no compatible checkpoint found under {root} for Fight Caves"
        )
    checkpoint = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    marker = find_checkpoint_marker(checkpoint, root)
    return {
        "resolved_path": str(checkpoint),
        "sidecar_path": marker,
        "format": checkpoint_format(checkpoint, expected_bytes),
    }
