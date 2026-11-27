"""Which client addresses may open a live page. The pages have no login, so a page that listens on all interfaces (``--public``) only answers
clients from these networks: this machine, the campus 10.0.0.0/8 and the Tailscale range 100.64.0.0/10. Others get HTTP 403.
Override with ``--allow 10.24.0.0/16 192.168.1.0/24`` (one or more CIDR networks) on the live commands."""
from __future__ import annotations

import ipaddress
from typing import Iterable

DEFAULT_ALLOW = ("127.0.0.0/8", "::1/128", "10.0.0.0/8", "100.64.0.0/10")


def parse_networks(nets: Iterable[str]) -> list:
    return [ipaddress.ip_network(n, strict=False) for n in nets]


def allowed(addr: str, nets: Iterable = DEFAULT_ALLOW) -> bool:
    """True if ``addr`` (a client IP, IPv4, IPv6 or IPv4-mapped IPv6) lies in one of the networks (strings or network objects)."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return any(ip in (n if not isinstance(n, str) else ipaddress.ip_network(n, strict=False)) for n in nets)
    except ValueError:
        return False
