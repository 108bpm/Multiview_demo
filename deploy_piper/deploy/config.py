"""Configuration loading for the split server/client deploy commands. stdlib-only.

A config (deploy/configs/<name>.json) records the deployment flags each side
needs. Activate the appropriate environment before running either command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Client config keys that build_client_args handles specially; everything else
# is forwarded verbatim as --key=value (e.g. first_predict_timeout_s)
CLIENT_STRUCTURAL_KEYS = {"robot_type", "robot_id", "cameras", "camera_map", "server"}


def load_config(name_or_path: str, configs_dir: Path = CONFIGS_DIR) -> dict:
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = configs_dir / f"{name_or_path}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in configs_dir.glob("*.json")))
        raise SystemExit(f"unknown config '{name_or_path}'. Available: {available}")
    config = json.loads(path.read_text())
    server = config.get("server")
    if not isinstance(server, dict):
        raise SystemExit(f"{path}: config needs a 'server' section")
    missing = [key for key in ("adapter", "port") if key not in server]
    if missing:
        raise SystemExit(f"{path}: server section is missing {', '.join(missing)}")
    if not isinstance(server["port"], int) or not 1 <= server["port"] <= 65535:
        raise SystemExit(f"{path}: server.port must be an integer from 1 to 65535")
    client = config.get("client")
    if client is not None:
        if not isinstance(client, dict):
            raise SystemExit(f"{path}: client must be an object")
        missing = [key for key in ("robot_type", "robot_id", "cameras") if key not in client]
        if missing:
            raise SystemExit(f"{path}: client section is missing {', '.join(missing)}")
        if not isinstance(client["cameras"], dict):
            raise SystemExit(f"{path}: client.cameras must be an object")
        if not isinstance(client.get("camera_map", {}), dict):
            raise SystemExit(f"{path}: client.camera_map must be an object")
    return config


def resolve_path(value: str) -> str:
    """Resolve config paths against the repo root; leave hub ids untouched."""
    candidate = REPO_ROOT / value
    return str(candidate.resolve()) if candidate.exists() else value


def extract_flag(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    """Pull --name=value or --name value out of argv; return (value, rest)."""
    value = None
    rest = []
    i = 0
    prefix = f"--{name}="
    flag = f"--{name}"
    while i < len(argv):
        item = argv[i]
        if item.startswith(prefix):
            value = item[len(prefix):]
        elif item == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} needs a value")
            value = argv[i + 1]
            i += 1
        else:
            rest.append(item)
        i += 1
    return value, rest


def build_server_args(config: dict) -> list[str]:
    server = config["server"]
    args = [
        f"--adapter={server['adapter']}",
        # 0.0.0.0 exposes the server to the LAN for a remote robot client
        f"--host={server.get('host', '127.0.0.1')}",
        f"--port={server['port']}",
    ]
    for key, value in server.get("args", {}).items():
        if key == "checkpoint":
            value = resolve_path(str(value))
        args.append(f"--{key}={value}")
    return args


def build_client_args(config: dict, task: str, extra_flags: list[str]) -> list[str]:
    client = config["client"]
    # client.server points at a remote policy server; default is same-machine
    server_url = client.get("server") or f"http://127.0.0.1:{config['server']['port']}"
    args = [
        f"--robot.type={client['robot_type']}",
        f"--robot.id={client['robot_id']}",
        f"--robot.cameras={json.dumps(client['cameras'])}",
        f"--server={server_url}",
        f"--task={task}",
        f"--camera_map={json.dumps(client.get('camera_map', {}))}",
    ]
    for key, value in client.items():
        if key not in CLIENT_STRUCTURAL_KEYS:
            args.append(f"--{key}={value}")
    args += extra_flags  # last value wins in draccus, so CLI overrides config
    return args


def bootstrap(section: str, argv: list[str] | None = None) -> list[str]:
    """Apply --config for deploy.server ("server") / deploy.client ("client").

    Without --config, argv is returned unchanged. With one, the config's
    section is expanded into command-line flags. The active Python environment
    is never changed. CLI flags come after config flags, so they win.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    config_name, rest = extract_flag(argv, "config")
    if not config_name:
        return argv
    config = load_config(config_name)
    if section == "client":
        if "client" not in config:
            raise SystemExit(f"config '{config_name}' has no client section")
        task, rest = extract_flag(rest, "task")
        return build_client_args(config, task or "", rest)
    else:
        return build_server_args(config) + rest
