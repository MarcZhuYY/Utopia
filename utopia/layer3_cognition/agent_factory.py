"""Agent Factory - Factory pattern for creating Agent Personas.

Provides AgentFactory.create_agent() to instantiate Persona subclasses
based on role, with support for parameter override through kwargs.

Example:
    >>> # Create standard retail investor
    >>> retail = AgentFactory.create_agent("retail", "RETAIL_001")
    >>>
    >>> # Create slightly calmer retail (neuroticism=0.7 instead of default 0.9)
    >>> calm_retail = AgentFactory.create_agent(
    ...     "retail", "RETAIL_002", neuroticism=0.7
    ... )
    >>>
    >>> # Create national team regulator
    >>> regulator = AgentFactory.create_agent("regulator", "CSRC_001")
"""

from __future__ import annotations

from typing import Type

from pydantic import ValidationError

from utopia.layer3_cognition.agent_persona_models import (
    AgentRole,
    BaseAgentPersona,
    InsiderPersona,
    MacroRegulatorPersona,
    QuantInstitutionPersona,
    RetailInvestorPersona,
)


class AgentFactory:
    """
    Agent instantiation factory.

    Automatically instantiates the appropriate Persona subclass based on role,
    allowing parameter fine-tuning through kwargs (still subject to Pydantic validation).
    """

    _ROLE_MAP: dict[AgentRole, Type[BaseAgentPersona]] = {
        AgentRole.RETAIL: RetailInvestorPersona,
        AgentRole.QUANT: QuantInstitutionPersona,
        AgentRole.INSIDER: InsiderPersona,
        AgentRole.REGULATOR: MacroRegulatorPersona,
    }

    @classmethod
    def create_agent(
        cls,
        role: str | AgentRole,
        agent_id: str,
        capital_weight: float | None = None,
        **kwargs,
    ) -> BaseAgentPersona:
        """
        Create Agent Persona based on role.

        Args:
            role: Role type ("retail", "quant", "insider", "regulator")
            agent_id: Unique agent identifier
            capital_weight: Capital weight (None uses default value)
            **kwargs: Can override default parameters (openness, neuroticism, etc.)

        Returns:
            Instantiated Persona object

        Raises:
            ValueError: If role is invalid
            ValidationError: If parameters are out of bounds

        Example:
            >>> retail = AgentFactory.create_agent("retail", "RETAIL_001")
            >>> retail.openness
            0.8
            >>> calm = AgentFactory.create_agent("retail", "R002", neuroticism=0.7)
            >>> calm.neuroticism
            0.7
        """
        # Normalize role
        if isinstance(role, str):
            try:
                role_enum = AgentRole(role.lower())
            except ValueError:
                valid_roles = [r.value for r in AgentRole]
                raise ValueError(
                    f"Invalid role: {role}. Valid roles: {valid_roles}"
                ) from None
        else:
            role_enum = role

        # Get corresponding class
        persona_class = cls._ROLE_MAP[role_enum]

        # Prepare constructor arguments
        init_kwargs: dict = {"agent_id": agent_id}

        # Special handling for capital_weight (Regulator forced to 10000.0)
        if capital_weight is not None and role_enum != AgentRole.REGULATOR:
            init_kwargs["capital_weight"] = capital_weight

        # Merge other parameters (allows overriding defaults)
        init_kwargs.update(kwargs)

        # Instantiate (Pydantic auto-validates)
        return persona_class(**init_kwargs)

    @classmethod
    def create_batch(
        cls,
        role: str | AgentRole,
        count: int,
        id_prefix: str,
        **kwargs,
    ) -> list[BaseAgentPersona]:
        """
        Batch create Agents.

        Args:
            role: Role type
            count: Number to create
            id_prefix: ID prefix (e.g., "RETAIL" generates RETAIL_001, RETAIL_002...)
            **kwargs: Parameters passed to create_agent

        Returns:
            List of Persona objects

        Example:
            >>> retailers = AgentFactory.create_batch("retail", 70, "RETAIL")
            >>> len(retailers)
            70
            >>> retailers[0].agent_id
            'RETAIL_001'
        """
        agents: list[BaseAgentPersona] = []
        for i in range(count):
            agent_id = f"{id_prefix}_{i+1:03d}"
            agent = cls.create_agent(role, agent_id, **kwargs)
            agents.append(agent)
        return agents

    @classmethod
    def get_default_params(cls, role: str | AgentRole) -> dict:
        """
        Get default parameters for a role.

        Args:
            role: Role type

        Returns:
            Dictionary of default parameters

        Example:
            >>> AgentFactory.get_default_params("retail")
            {'openness': 0.8, 'neuroticism': 0.9, ...}
        """
        if isinstance(role, str):
            role = AgentRole(role.lower())

        persona_class = cls._ROLE_MAP[role]
        # Create temporary instance to get defaults
        temp = persona_class(agent_id="temp", capital_weight=1.0)
        return {
            "openness": temp.openness,
            "neuroticism": temp.neuroticism,
            "conscientiousness": temp.conscientiousness,
            "influence_weight": temp.influence_weight,
            "capital_weight": (
                10000.0 if role == AgentRole.REGULATOR else temp.capital_weight
            ),
        }

    @classmethod
    def get_role_class(cls, role: str | AgentRole) -> Type[BaseAgentPersona]:
        """
        Get the Persona class for a role.

        Args:
            role: Role type

        Returns:
            Persona class
        """
        if isinstance(role, str):
            role = AgentRole(role.lower())
        return cls._ROLE_MAP[role]
