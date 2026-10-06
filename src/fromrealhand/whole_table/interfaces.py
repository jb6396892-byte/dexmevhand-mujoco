"""Implementation ports for later milestones; none move the simulator yet."""
from abc import ABC, abstractmethod
from typing import Tuple
from .contracts import CompositeCommand, PlanningResult, TaskRequest, Telemetry


class SceneBackend(ABC):
    @abstractmethod
    def reset(self, request: TaskRequest) -> Telemetry:
        """Build/settle an initial scene; pose writes are allowed only here."""
        raise NotImplementedError

    @abstractmethod
    def step(self, command: CompositeCommand) -> Telemetry:
        """Apply named actuator controls and integrate physics, without pose writes."""
        raise NotImplementedError


class TransitPlanner(ABC):
    @abstractmethod
    def plan(self, request: TaskRequest, state: Telemetry) -> PlanningResult:
        """Check whole hand, carriage, payload and swept motions on a planning copy."""
        raise NotImplementedError


class LocalPolicyAdapter(ABC):
    @abstractmethod
    def action(self, local_features: Tuple[float, ...],
               reference_action: Tuple[float, ...], video: str, skill: str) -> Tuple[float, ...]:
        """Translate the legacy 139-feature/30-action policy through a local frame."""
        raise NotImplementedError
