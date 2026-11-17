"""Project aerodynamic force models (decision D5) and their injection into the simulator.

Gravity and rotor thrust / reaction torque are simulator-native (PhysX / OmniDrones)
and are not modelled here (docs/models/force_accounting.md).
"""
from .injection import InjectionError, WrenchInjector, install
from .registry import REGISTRY, ModelSpec
from .system import COMPONENTS, AeroForceSystem, WrenchResult

__all__ = ["AeroForceSystem", "COMPONENTS", "InjectionError", "ModelSpec", "REGISTRY",
           "WrenchInjector", "WrenchResult", "install"]
