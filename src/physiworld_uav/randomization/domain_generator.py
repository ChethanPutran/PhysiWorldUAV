"""Generate the domains of one experiment split role (plan §6.4).

OOD domains are drawn from a seed branch disjoint from the ID branch
(``seed_node.child("region:<region>")``) so that adding or removing OOD
domains never changes the training domains, and vice versa.
"""
from __future__ import annotations

from typing import Any, Mapping

from physiworld_uav.core.seeds import SeedNode
from physiworld_uav.drones.params import DroneParams

from .parameter_sampler import Domain, DomainSamplerConfig, nominal_domain, sample_domains


def generate_domains(
    domain_cfg: Mapping[str, Any],
    nominal: DroneParams,
    n_domains: int,
    seed_node: SeedNode,
    *,
    id_prefix: str,
) -> list[Domain]:
    """``domain_cfg`` is the composed ``domain`` config group."""
    region = str(domain_cfg.get("region", "nominal"))
    if region == "nominal":
        return [nominal_domain()]
    cfg = DomainSamplerConfig.from_dict(domain_cfg)
    rng = seed_node.child(f"region:{region}").rng()
    doms = sample_domains(cfg, nominal, int(n_domains), region, rng, id_prefix=id_prefix)
    return [
        Domain(domain_id=d.domain_id, region=d.region, factors=d.factors,
               coordinates=d.coordinates,
               seed_path=seed_node.child(f"region:{region}").path_str)
        for d in doms
    ]
