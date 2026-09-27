import pytest
from fastapi.testclient import TestClient
import main

@pytest.fixture
def client():
    # Reset limiter memory between tests if present
    if hasattr(main, "rate_limiter"):
        main.rate_limiter.reset()
    return TestClient(main.app)

def test_security_headers_present(client):
    """Test that all HTTP responses include OWASP recommended security headers."""
    response = client.get("/api/health")
    assert response.status_code == 200
    headers = {k.lower(): v for k, v in response.headers.items()}
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert "strict-origin-when-cross-origin" in headers.get("referrer-policy", "")
    assert "content-security-policy" in headers
    assert "permissions-policy" in headers

def test_payload_too_large(client):
    """Test that oversized payload (> 64 KB) is rejected with 413 Payload Too Large."""
    # Create JSON payload with extra padding > 64 KB
    padded_payload = '{"smiles": "CCO", "concentration_molar": 1.0, "padding": "' + ("A" * 70000) + '"}'
    response = client.post(
        "/api/predict",
        content=padded_payload,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert "Payload Too Large" in response.text or "too large" in response.text.lower()

def test_rate_limiting_predict_endpoint(client):
    """Test that rapid requests to /api/predict trigger rate limiting (429) with Retry-After header."""
    # Send requests up to rate limit threshold
    limit = getattr(main, "PREDICT_RATE_LIMIT", 20)
    hit_429 = False
    
    for i in range(limit + 5):
        resp = client.post(
            "/api/predict",
            json={"smiles": "CCO", "concentration_molar": 1.0}
        )
        if resp.status_code == 429:
            hit_429 = True
            assert "retry-after" in resp.headers
            assert int(resp.headers["retry-after"]) >= 1
            data = resp.json()
            assert "detail" in data
            assert "Rate limit" in data["detail"] or "rate limit" in data["detail"].lower()
            break
            
    assert hit_429, f"Expected 429 Too Many Requests within {limit + 5} requests"

def test_health_check_exempt_from_rate_limiting(client):
    """Test that health checks are exempt from rate limiting for reliable orchestrator probes."""
    for _ in range(50):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

def test_smiles_length_limit(client):
    """Test that SMILES longer than 500 characters are rejected with 422 Unprocessable Entity."""
    long_smiles = "C" * 501
    resp = client.post(
        "/api/predict",
        json={"smiles": long_smiles, "concentration_molar": 1.0}
    )
    assert resp.status_code == 422

def test_concentration_range_validation(client):
    """Test that invalid concentration values (< 1e-12 or > 10.0 M) are rejected with 422."""
    # Negative concentration
    resp_neg = client.post(
        "/api/predict",
        json={"smiles": "CCO", "concentration_molar": -1.0}
    )
    assert resp_neg.status_code == 422

    # Excessively high concentration (> 10.0 M)
    resp_high = client.post(
        "/api/predict",
        json={"smiles": "CCO", "concentration_molar": 50.0}
    )
    assert resp_high.status_code == 422

def test_safe_error_handling_invalid_smiles(client):
    """Test that invalid SMILES returns 400 Bad Request without leaking internal stack traces."""
    resp = client.post(
        "/api/predict",
        json={"smiles": "INVALID_CHEM_!@#$%", "concentration_molar": 1.0}
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "Invalid SMILES" in data.get("detail", "")
    # Ensure no python traceback is leaked in body
    assert "Traceback (most recent call last)" not in resp.text
    assert "File \"" not in resp.text
