"""Doctor uses the real AWS role credential chain against loopback services (#153434)."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli import config, doctor, doctor_connectivity


@pytest.fixture
def aws_role_home(tmp_path, monkeypatch):
    boto3 = pytest.importorskip("boto3", reason="Bedrock extra required for the real AWS credential chain")
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    aws_config = tmp_path / "aws-config"
    aws_config.write_text("[default]\nregion=us-east-1\n[profile role]\nregion=us-east-1\n", encoding="utf-8")
    credentials = tmp_path / "credentials"
    credentials.write_text("", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(aws_config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_PUT(self):
            requests.append(self.path)
            assert self.path == "/latest/api/token"
            self.reply(b"fixture-token")

        def do_GET(self):
            requests.append(self.path)
            if self.path == "/latest/meta-data/iam/security-credentials/":
                self.reply(b"fixture-role")
            elif self.path == "/latest/meta-data/iam/security-credentials/fixture-role":
                self.reply(json.dumps({
                    "Code": "Success", "AccessKeyId": "fixture-access", "SecretAccessKey": "fixture-secret",
                    "Token": "fixture-session", "Expiration": "2035-01-01T00:00:00Z",
                }).encode())
            elif self.path == "/foundation-models":
                assert "fixture-access" in self.headers["Authorization"]
                self.reply(b'{"modelSummaries": []}')
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", endpoint)
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", endpoint)
    monkeypatch.setattr(doctor, "build_probes", lambda: [("AWS Bedrock", doctor_connectivity._probe_bedrock)])
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", (("API Connectivity", doctor._check_api_connectivity),))
    config._LOAD_CONFIG_CACHE.clear()
    try:
        yield home, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        config._LOAD_CONFIG_CACHE.clear()


@pytest.mark.parametrize("provider,named_profile", [("bedrock", False), ("aws-bedrock", False), ("openai", True)])
def test_doctor_reports_instance_role_healthy(aws_role_home, monkeypatch, capsys, provider, named_profile):
    home, requests = aws_role_home
    (home / "config.yaml").write_text(f"model:\n  provider: {provider}\n", encoding="utf-8")
    if named_profile:
        monkeypatch.setenv("AWS_PROFILE", "role")
    assert doctor.run_doctor(SimpleNamespace(fix=False, live=False)) == 0
    output = capsys.readouterr().out
    assert "AWS Bedrock" in output and "0 models" in output
    assert "All checks passed" in output
    assert "/latest/meta-data/iam/security-credentials/fixture-role" in requests
    assert "/foundation-models" in requests
    assert "AWS_EC2_METADATA_DISABLED" not in os.environ


@pytest.mark.parametrize("explicit_opt_out", [False, True])
def test_doctor_preserves_no_aws_and_explicit_imds_opt_out(aws_role_home, monkeypatch, explicit_opt_out):
    home, requests = aws_role_home
    provider = "bedrock" if explicit_opt_out else "openai"
    (home / "config.yaml").write_text(f"model:\n  provider: {provider}\n", encoding="utf-8")
    if explicit_opt_out:
        monkeypatch.setenv("AWS_PROFILE", "role")
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    finding = doctor._check_api_connectivity(False)
    assert requests == []
    if explicit_opt_out:
        assert any("NoCredentialsError" in issue for issue in finding.issues)
        assert os.environ["AWS_EC2_METADATA_DISABLED"] == "true"
    else:
        assert finding.issues == []
        assert "AWS_EC2_METADATA_DISABLED" not in os.environ
