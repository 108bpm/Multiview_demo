import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from deploy.config import build_client_args, load_config
from deploy.protocol import PROTOCOL_VERSION, PolicyClient, ProtocolMismatchError


class ConfigTests(unittest.TestCase):
    def test_client_config_expands_to_flags_only(self):
        config = {
            "server": {"adapter": "dummy", "port": 8090},
            "client": {"robot_type": "robot", "robot_id": "id", "cameras": {}},
        }
        args = build_client_args(config, "home", [])
        self.assertIn("--robot.type=robot", args)
        self.assertIn("--task=home", args)
        self.assertFalse(any("python" in arg.lower() for arg in args))

    def test_invalid_port_fails_while_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bad.json")
            path.write_text(json.dumps({
                "server": {"adapter": "dummy", "port": 70000}
            }))
            with self.assertRaisesRegex(SystemExit, "server.port"):
                load_config(str(path))


class ProtocolTests(unittest.TestCase):
    def test_policy_client_rejects_incompatible_server(self):
        with patch("deploy.protocol.http_get_json", return_value={"protocol_version": 99}):
            with self.assertRaises(ProtocolMismatchError):
                PolicyClient("http://server").info()

    def test_policy_client_accepts_current_protocol(self):
        expected = {"protocol_version": PROTOCOL_VERSION, "fps": 30}
        with patch("deploy.protocol.http_get_json", return_value=expected):
            self.assertEqual(PolicyClient("http://server/").info(), expected)


class ActionSafetyTests(unittest.TestCase):
    def test_non_finite_action_is_rejected(self):
        from deploy.client import action_to_dict

        with self.assertRaisesRegex(ValueError, "non-finite"):
            action_to_dict(np.array([0.0, np.nan]), ["a", "b"])

    def test_wrong_action_size_is_rejected(self):
        from deploy.client import action_to_dict

        with self.assertRaisesRegex(ValueError, "expected"):
            action_to_dict(np.array([0.0]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
