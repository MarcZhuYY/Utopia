"""Rule engine for validating agent actions.

Implements three types of rules:
1. Physical rules (hard constraints)
2. Social role rules (soft constraints)
3. Domain-specific rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from utopia.core.config import DOMAIN_RULES, ROLE_CONSTRAINTS, WorldRules
from utopia.core.models import Action, Entity
from utopia.layer3_cognition.agent import Agent


@dataclass
class ValidationResult:
    """Result of action validation.

    Attributes:
        allowed: Whether action is allowed
        reason: Reason if not allowed
        warning: Warning if soft constraint violated
    """

    allowed: bool = True
    reason: str = ""
    warning: str = ""


class RuleEngine:
    """Engine for validating actions against world rules.

    This is a stateless validator - all state is passed in via context.
    """

    def __init__(self, world_rules: Optional[WorldRules] = None, domain: str = "general"):
        """Initialize rule engine.

        Args:
            world_rules: World rules to enforce
            domain: Domain context (general/financial/political)
        """
        self.world_rules = world_rules or WorldRules()
        self.domain = domain
        self.domain_rules = DOMAIN_RULES.get(domain, DOMAIN_RULES["general"])

    def validate_action(
        self,
        agent: Agent,
        action: Action,
        world_context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate an action against all rule types.

        Args:
            agent: Agent attempting the action
            action: Action to validate
            world_context: Optional world state context

        Returns:
            ValidationResult: Validation result
        """
        # 1. Check physical rules
        physical_result = self._check_physical_rules(action)
        if not physical_result.allowed:
            return physical_result

        # 2. Check social role rules (soft constraint)
        role_result = self._check_role_constraints(agent, action)
        if role_result.warning:
            # Soft constraint: allowed but with warning
            pass

        # 3. Check domain rules
        domain_result = self._check_domain_rules(agent, action, world_context)
        if not domain_result.allowed:
            return domain_result

        # Return combined result
        if role_result.warning:
            return ValidationResult(
                allowed=True,
                warning=f"Soft constraint: {role_result.warning}",
            )

        return ValidationResult(allowed=True)

    def _check_physical_rules(self, action: Action) -> ValidationResult:
        """Check physical/hard constraints.

        Args:
            action: Action to check

        Returns:
            ValidationResult: Hard constraint check result
        """
        # Rule: max_actions_per_tick already enforced by engine

        # Rule: action content must not be empty (except 'silent')
        if action.action_type != "silent":
            if not action.content or len(action.content.strip()) == 0:
                return ValidationResult(
                    allowed=False,
                    reason="Action content cannot be empty",
                )

        # Rule: action type must be valid
        valid_types = {"speak", "private_message", "change_belief", "act", "silent"}
        if action.action_type not in valid_types:
            return ValidationResult(
                allowed=False,
                reason=f"Invalid action type: {action.action_type}",
            )

        return ValidationResult(allowed=True)

    def _check_role_constraints(self, agent: Agent, action: Action) -> ValidationResult:
        """Check social role constraints (soft).

        Args:
            agent: Agent attempting the action
            action: Action to check

        Returns:
            ValidationResult: Soft constraint check result
        """
        role = agent.persona.role
        role_constraints = ROLE_CONSTRAINTS.get(role, {})

        # Check if role allows public speaking
        if action.action_type == "speak":
            if not role_constraints.get("can_speak_publicly", True):
                return ValidationResult(
                    allowed=True,
                    warning=f"Role '{role}' typically cannot speak publicly",
                )

        # Check if action is in typical topics
        typical_topics = role_constraints.get("typical_topics", ["all"])
        if "all" not in typical_topics and action.topic_id:
            # Check if topic matches role expertise
            if action.topic_id not in typical_topics:
                return ValidationResult(
                    allowed=True,
                    warning=f"Role '{role}' typically doesn't engage with topic '{action.topic_id}'",
                )

        return ValidationResult(allowed=True)

    def _check_domain_rules(
        self,
        agent: Agent,
        action: Action,
        world_context: Optional[dict[str, Any]],
    ) -> ValidationResult:
        """Check domain-specific rules.

        Args:
            agent: Agent attempting the action
            action: Action to check
            world_context: World state context

        Returns:
            ValidationResult: Domain rule check result
        """
        if not self.domain_rules:
            return ValidationResult(allowed=True)

        # Financial domain rules
        if self.domain == "financial":
            return self._check_financial_rules(agent, action, world_context)

        # Political domain rules
        if self.domain == "political":
            return self._check_political_rules(agent, action, world_context)

        return ValidationResult(allowed=True)

    def _check_financial_rules(
        self,
        agent: Agent,
        action: Action,
        world_context: Optional[dict[str, Any]],
    ) -> ValidationResult:
        """Check financial domain rules.

        Args:
            agent: Agent attempting the action
            action: Action to check
            world_context: World state context

        Returns:
            ValidationResult: Financial rule check result
        """
        # Check insider trading rule
        insider_rule = self.domain_rules.get("no_insider_trading", "")
        if insider_rule and action.action_type == "act":
            # Check if agent has insider information
            if world_context and world_context.get("has_insider_info", False):
                return ValidationResult(
                    allowed=False,
                    reason=f"Rule violation: {insider_rule}",
                )

        return ValidationResult(allowed=True)

    def _check_political_rules(
        self,
        agent: Agent,
        action: Action,
        world_context: Optional[dict[str, Any]],
    ) -> ValidationResult:
        """Check political domain rules.

        Args:
            agent: Agent attempting the action
            action: Action to check
            world_context: World state context

        Returns:
            ValidationResult: Political rule check result
        """
        # Check electoral cycle influence
        if world_context:
            election_near = world_context.get("election_near", False)
            if election_near and agent.persona.role == "politician":
                # Near elections, politicians tend toward more extreme positions
                # This is informational, not a constraint
                pass

        return ValidationResult(allowed=True)


class RuleValidator:
    """Static validator methods for convenience."""

    @staticmethod
    def validate_action(
        agent: Agent,
        action: Action,
        world_rules: Optional[WorldRules] = None,
        domain: str = "general",
        world_context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate an action using default engine.

        Args:
            agent: Agent attempting the action
            action: Action to validate
            world_rules: World rules
            domain: Domain context
            world_context: World state context

        Returns:
            ValidationResult: Validation result
        """
        engine = RuleEngine(world_rules=world_rules, domain=domain)
        return engine.validate_action(agent, action, world_context)
