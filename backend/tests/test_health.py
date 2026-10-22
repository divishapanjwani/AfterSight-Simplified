from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["scope"] == "Damage classification on building footprints derived from xBD annotations."


def test_cors_allows_only_vite_origin() -> None:
    allowed = client.get("/api/health", headers={"Origin": "http://localhost:5200"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5200"

    blocked = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers


def test_no_deep_learning_frameworks_installed() -> None:
    import importlib.util

    for forbidden in ("torch", "tensorflow", "keras"):
        assert importlib.util.find_spec(forbidden) is None, f"{forbidden} must not be installed"
