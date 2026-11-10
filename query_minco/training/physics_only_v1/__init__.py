"""Isolated S0 physical-cost mathematics; no teacher or training runner."""
from .spec import PhysicsSpec
from .objective import PhysicsOnlyObjective

__all__ = ["PhysicsSpec", "PhysicsOnlyObjective"]
