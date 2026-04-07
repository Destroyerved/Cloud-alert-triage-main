"""
server/service_graph.py

Defines a realistic microservice dependency graph for the Cloud Alert Triage
environment. The graph models a mid-size tech company with 17 services arranged
in a clear 5-tier hierarchy:

    Tier 1 (Client):       web-frontend
    Tier 2 (Gateway):      api-gateway
    Tier 3 (Core APIs):    auth-service, user-service, order-service,
                           search-service, notification-service
    Tier 4 (Workers /      payment-gateway, inventory-service,
             Integrations): recommendation-engine, email-worker, sms-worker
    Tier 5 (Data Layer):   postgres-primary, redis-cache, kafka-broker,
                           elasticsearch, object-storage

Edges represent "depends on" relationships (i.e. A → B means A calls B).
Leaf nodes (no dependencies) have an empty list.

This module is intentionally free of external imports so it can be used by
both server code and tests without any side effects.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core graph definition
# ---------------------------------------------------------------------------
# Dict[service_name, List[dependency_name]]
# A service's entry lists the services it directly calls / depends on.
SERVICE_GRAPH: dict[str, list[str]] = {
    # ── Tier 1: client-facing ──────────────────────────────────────────────
    "web-frontend": ["api-gateway"],

    # ── Tier 2: API gateway ────────────────────────────────────────────────
    "api-gateway": [
        "auth-service",
        "user-service",
        "order-service",
        "search-service",
        "notification-service",
    ],

    # ── Tier 3: core business services ────────────────────────────────────
    "auth-service": ["postgres-primary", "redis-cache"],

    "user-service": ["postgres-primary", "redis-cache"],

    "order-service": [
        "postgres-primary",
        "payment-gateway",
        "inventory-service",
        "kafka-broker",
    ],

    "search-service": ["elasticsearch"],

    "notification-service": ["email-worker", "sms-worker", "kafka-broker"],

    # ── Tier 4: workers and third-party integrations ───────────────────────
    "payment-gateway": ["postgres-primary"],

    "inventory-service": ["postgres-primary"],

    "recommendation-engine": ["elasticsearch", "redis-cache"],

    "email-worker": ["object-storage"],

    "sms-worker": [],  # external SMS provider — no internal deps

    # ── Tier 5: data / infrastructure layer (leaf nodes) ──────────────────
    "postgres-primary": [],
    "redis-cache": [],
    "kafka-broker": [],
    "elasticsearch": [],
    "object-storage": [],
}

# ---------------------------------------------------------------------------
# Pre-computed reverse index: dependents (who calls me?)
# ---------------------------------------------------------------------------
_DEPENDENTS: dict[str, list[str]] = {svc: [] for svc in SERVICE_GRAPH}
for _svc, _deps in SERVICE_GRAPH.items():
    for _dep in _deps:
        _DEPENDENTS[_dep].append(_svc)

# Sort for determinism
for _svc in _DEPENDENTS:
    _DEPENDENTS[_svc].sort()


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def get_service_names() -> list[str]:
    """Return a sorted list of all service names in the graph."""
    return sorted(SERVICE_GRAPH.keys())


def get_dependencies(service: str) -> list[str]:
    """
    Return the list of services that *service* directly depends on
    (i.e. the services it calls).

    Returns an empty list for leaf nodes or unknown service names.
    """
    return list(SERVICE_GRAPH.get(service, []))


def get_dependents(service: str) -> list[str]:
    """
    Return the list of services that directly depend on *service*
    (i.e. the services that call it).

    Returns an empty list for root nodes or unknown service names.
    """
    return list(_DEPENDENTS.get(service, []))


def get_graph_as_adjacency_list() -> dict[str, list[str]]:
    """
    Return the full dependency graph as a plain dict suitable for JSON
    serialisation and inclusion in the Observation payload.

    Keys are sorted alphabetically for determinism.
    """
    return {svc: list(deps) for svc, deps in sorted(SERVICE_GRAPH.items())}
