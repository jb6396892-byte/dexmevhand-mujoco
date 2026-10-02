"""Finite Chinese aliases map to safe symbolic plans, not free-form actions."""
from ..skills import SKILLS, safety_failure
from ..skill_pipeline import GuardedSkill
from .registry import normalize_instruction


class RulePlanner:
    def __init__(self, registry):
        self.registry = registry

    def plan(self, instruction, scene='first'):
        key = normalize_instruction(instruction)
        if key not in self.registry.aliases:
            raise ValueError('Unsupported instruction; no action will be executed')
        goal = self.registry.aliases[key]
        skills = [] if goal == 'stop' else list(SKILLS[:SKILLS.index(goal)+1])
        return self.registry.validate_plan(dict(schema_version=1, scene=scene, object='mug',
                                                goal=goal, skills=skills))

    def replan(self, plan, completed, row, contract_config, cursor, bounds):
        plan = self.registry.validate_plan(plan)
        if completed != plan['skills'][:len(completed)]:
            raise ValueError('Completed skills do not match the original plan')
        remaining = plan['skills'][len(completed):]
        if safety_failure(row, contract_config):
            raise ValueError('Unsafe state cannot be replanned')
        if remaining:
            skill = remaining[0]
            if cursor != bounds[skill][0]:
                raise ValueError('Reference clock cannot rewind or jump during replanning')
            if GuardedSkill(skill, contract_config, row).status == 'failed':
                raise ValueError('No verified recovery path from this entry')
        return remaining
