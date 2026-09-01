"""Token-Bucket & Session Rate Limiter for SaaS Monetization."""

import time
from threading import Lock
from typing import NamedTuple
from fastapi import Request, HTTPException, status


class ClientUsage(NamedTuple):
    queries_used: int
    queries_limit: int
    queries_remaining: int
    total_tokens_consumed: int
    tier: str
    reset_in_seconds: int


class RateLimiter:
    """
    In-memory thread-safe rate limiter supporting Free and Pro subscription tiers.
    - Free Tier: 5 queries per session / IP window
    - Pro Tier: Unlimited queries
    """

    FREE_TIER_LIMIT = 5
    WINDOW_SECONDS = 3600  # 1 hour window

    def __init__(self):
        self._lock = Lock()
        # client_id -> {"count": int, "window_start": float, "tokens": int}
        self._clients: dict[str, dict] = {}

    def get_client_id_and_tier(self, request: Request) -> tuple[str, str]:
        """Resolves client identifier and tier from headers or IP address."""
        # 1. Tier Resolution: Header 'X-Subscription-Tier' or API Key prefix
        api_key = request.headers.get("X-API-Key", "")
        tier_header = request.headers.get("X-Subscription-Tier", "").lower().strip()

        tier = "free"
        if tier_header == "pro" or api_key.startswith("rm_pro_"):
            tier = "pro"

        # 2. Client Identifier Resolution: X-Session-ID > X-API-Key > Client Host
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            client_id = f"session_{session_id}"
        elif api_key:
            client_id = f"key_{api_key[:12]}"
        else:
            client_ip = request.client.host if request.client else "127.0.0.1"
            client_id = f"ip_{client_ip}"

        return client_id, tier

    def check_rate_limit(self, request: Request, estimated_tokens: int = 100) -> ClientUsage:
        """
        Validates rate limit for incoming request.
        Raises HTTP 429 if the Free Tier quota is exceeded.
        """
        client_id, tier = self.get_client_id_and_tier(request)
        now = time.time()

        with self._lock:
            client_data = self._clients.get(client_id)
            if not client_data or (now - client_data["window_start"]) > self.WINDOW_SECONDS:
                client_data = {
                    "count": 0,
                    "window_start": now,
                    "tokens": 0,
                }
                self._clients[client_id] = client_data

            if tier == "free":
                if client_data["count"] >= self.FREE_TIER_LIMIT:
                    time_remaining = int(self.WINDOW_SECONDS - (now - client_data["window_start"]))
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Rate limit exceeded",
                            "message": f"Free tier quota ({self.FREE_TIER_LIMIT} queries/hour) reached. Upgrade to Pro for unlimited Deep Research.",
                            "tier": "free",
                            "queries_limit": self.FREE_TIER_LIMIT,
                            "queries_remaining": 0,
                            "retry_after_seconds": max(1, time_remaining),
                            "upgrade_url": "https://researchmind.ai/pricing",
                        },
                        headers={
                            "X-RateLimit-Limit": str(self.FREE_TIER_LIMIT),
                            "X-RateLimit-Remaining": "0",
                            "Retry-After": str(max(1, time_remaining)),
                        },
                    )

            # Increment query counter and record estimated tokens
            client_data["count"] += 1
            client_data["tokens"] += estimated_tokens

            remaining = max(0, self.FREE_TIER_LIMIT - client_data["count"]) if tier == "free" else 999999
            limit = self.FREE_TIER_LIMIT if tier == "free" else -1
            reset_in = int(self.WINDOW_SECONDS - (now - client_data["window_start"]))

            return ClientUsage(
                queries_used=client_data["count"],
                queries_limit=limit,
                queries_remaining=remaining,
                total_tokens_consumed=client_data["tokens"],
                tier=tier,
                reset_in_seconds=max(0, reset_in),
            )

    def get_usage(self, request: Request) -> ClientUsage:
        """Retrieves current usage statistics without incrementing counters."""
        client_id, tier = self.get_client_id_and_tier(request)
        now = time.time()

        with self._lock:
            client_data = self._clients.get(client_id)
            if not client_data or (now - client_data["window_start"]) > self.WINDOW_SECONDS:
                count = 0
                tokens = 0
                reset_in = self.WINDOW_SECONDS
            else:
                count = client_data["count"]
                tokens = client_data["tokens"]
                reset_in = int(self.WINDOW_SECONDS - (now - client_data["window_start"]))

            remaining = max(0, self.FREE_TIER_LIMIT - count) if tier == "free" else 999999
            limit = self.FREE_TIER_LIMIT if tier == "free" else -1

            return ClientUsage(
                queries_used=count,
                queries_limit=limit,
                queries_remaining=remaining,
                total_tokens_consumed=tokens,
                tier=tier,
                reset_in_seconds=max(0, reset_in),
            )

    def reset(self) -> None:
        """Resets all usage data (useful for testing)."""
        with self._lock:
            self._clients.clear()


rate_limiter = RateLimiter()
