import time
import math
import threading
from collections import deque
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

DEFAULT_RATE_LIMIT = 120
PREDICT_RATE_LIMIT = 20
MAX_BODY_SIZE_BYTES = 64 * 1024  # 64 KB

class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter per client IP.
    Protects compute-heavy GNN endpoints from volumetric DoS and resource starvation.
    """
    def __init__(self, default_limit: int = DEFAULT_RATE_LIMIT, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.route_limits = {}  # prefix -> max requests per window
        self._history = {}       # (client_ip, bucket) -> deque([timestamp, ...])
        self._lock = threading.Lock()

    def set_route_limit(self, path_prefix: str, limit: int):
        self.route_limits[path_prefix] = limit

    def reset(self):
        """Reset all rate limiter tracking history."""
        with self._lock:
            self._history.clear()

    def check_request(self, client_ip: str, path: str) -> tuple[bool, int]:
        now = time.time()
        limit = self.default_limit
        bucket_key = "default"
        for prefix, route_lim in self.route_limits.items():
            if path.startswith(prefix):
                limit = route_lim
                bucket_key = prefix
                break

        key = (client_ip, bucket_key)
        cutoff = now - self.window_seconds

        with self._lock:
            if key not in self._history:
                self._history[key] = deque()
            
            queue = self._history[key]
            # Prune timestamps outside current sliding window
            while queue and queue[0] < cutoff:
                queue.popleft()

            if len(queue) >= limit:
                oldest = queue[0]
                retry_after = max(1, int(math.ceil(oldest + self.window_seconds - now)))
                return False, retry_after

            queue.append(now)
            return True, 0

rate_limiter = SlidingWindowRateLimiter(default_limit=DEFAULT_RATE_LIMIT, window_seconds=60)
rate_limiter.set_route_limit("/api/predict", PREDICT_RATE_LIMIT)

def get_client_ip(request: Request) -> str:
    """Extract client IP with proxy support (X-Forwarded-For, X-Real-IP)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces per-client IP sliding window rate limits with RFC 6585 Retry-After header."""
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Health probes and OpenAPI documentation are immune from rate limiting
        if path in {"/api/health", "/health", "/docs", "/openapi.json", "/favicon.ico"}:
            return await call_next(request)

        client_ip = get_client_ip(request)
        is_allowed, retry_after = rate_limiter.check_request(client_ip, path)
        if not is_allowed:
            limit_val = PREDICT_RATE_LIMIT if path.startswith("/api/predict") else DEFAULT_RATE_LIMIT
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Maximum {limit_val} requests per minute reached for this endpoint. Try again in {retry_after} seconds.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        return await call_next(request)

class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """Mitigates RAM exhaustion attacks by rejecting oversized payloads (>64KB)."""
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_BODY_SIZE_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload Too Large. Maximum allowed request body is 64 KB."}
                    )
            except ValueError:
                pass
        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attaches OWASP recommended security headers to all HTTP responses."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Permissions-Policy"] = "accelerometer=(), camera=(), geolocation=(), microphone=()"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
