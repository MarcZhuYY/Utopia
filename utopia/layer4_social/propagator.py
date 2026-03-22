"""Information propagator with cognitive distortion.

Implements BFS propagation with depth decay and receiver bias.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx

from utopia.core.config import WorldRules
from utopia.core.models import Message
from utopia.layer4_social.relationships import RelationshipMap

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


@dataclass
class ReceivedMessage:
    """Message as received by an agent.

    Attributes:
        message: Original message
        from_agent: Sender ID
        to_agent: Receiver ID
        depth: Propagation depth from original sender
        distortion_applied: Whether cognitive distortion was applied
        trust_at_reception: Trust level when received
    """

    message: Message
    from_agent: str = ""
    to_agent: str = ""
    depth: int = 0
    distortion_applied: bool = False
    trust_at_reception: float = 0.5


class InformationPropagator:
    """Propagates information through the agent network.

    Implements:
    - BFS propagation with depth decay
    - Trust-based reception probability
    - Cognitive distortion at receiver
    """

    def __init__(self, world_rules: Optional[WorldRules] = None):
        """Initialize propagator.

        Args:
            world_rules: World rules for propagation parameters
        """
        self.rules = world_rules or WorldRules()

    def propagate(
        self,
        message: Message,
        sender: Agent,
        graph: nx.DiGraph,
        relationships: RelationshipMap,
        agents: dict[str, Agent],
    ) -> list[ReceivedMessage]:
        """Propagate message through network.

        Args:
            message: Message to propagate
            sender: Sending agent
            graph: Network graph
            relationships: Relationship map
            agents: Dict of agent_id -> Agent

        Returns:
            list[ReceivedMessage]: List of received messages
        """
        received: list[ReceivedMessage] = []
        visited: set[str] = {sender.id}
        queue: deque[tuple[str, int]] = deque()

        # Initialize with direct neighbors
        try:
            for neighbor in graph.neighbors(sender.id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, 1))  # depth 1
        except nx.NetworkXError:
            return received

        max_depth = self.rules.max_propagation_depth
        decay = self.rules.propagation_decay

        while queue:
            current_id, depth = queue.popleft()

            if depth > max_depth:
                continue

            current_agent = agents.get(current_id)
            if not current_agent:
                continue

            edge = relationships.get(sender.id, current_id)

            # Calculate receive probability
            receive_prob = edge.trust * (decay ** (depth - 1))

            # Stochastic filtering
            import random
            if random.random() > receive_prob:
                continue

            # Get receiver's stance for distortion calculation
            receiver_stance = current_agent.get_stance(message.topic_id)
            original_stance = message.original_stance

            # Apply cognitive distortion if enabled
            distorted_content = message.content
            distortion_applied = False

            if self.rules.distortion_coefficient > 0:
                distorted_content, distortion_applied = self._apply_distortion(
                    content=message.content,
                    original_stance=original_stance,
                    receiver_stance=receiver_stance.position if receiver_stance else 0.0,
                    receiver_confidence=receiver_stance.confidence if receiver_stance else 0.5,
                    receiver_role=current_agent.persona.role,
                )

            # Create received message
            received_msg = Message(
                content=distorted_content,
                sender_id=message.sender_id,
                receiver_id=current_id,
                topic_id=message.topic_id,
                original_stance=message.original_stance,
                timestamp=datetime.now(),
            )

            received.append(ReceivedMessage(
                message=received_msg,
                from_agent=sender.id,
                to_agent=current_id,
                depth=depth,
                distortion_applied=distortion_applied,
                trust_at_reception=edge.trust * (decay ** (depth - 1)),
            ))

            # Continue propagation
            try:
                for neighbor in graph.neighbors(current_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            except nx.NetworkXError:
                continue

        return received

    def _apply_distortion(
        self,
        content: str,
        original_stance: float,
        receiver_stance: float,
        receiver_confidence: float,
        receiver_role: str,
    ) -> tuple[str, bool]:
        """Apply cognitive distortion to message.

        People tend to accept information that confirms their worldview
        and distort or filter out contradictory information.

        Args:
            content: Original content
            original_stance: Sender's stance
            receiver_stance: Receiver's stance
            receiver_confidence: Receiver's confidence
            receiver_role: Receiver's role

        Returns:
            tuple[str, bool]: (distorted content, whether distortion was applied)
        """
        stance_diff = abs(receiver_stance - original_stance)
        distortion_factor = stance_diff * self.rules.distortion_coefficient

        if distortion_factor < 0.05:
            return content, False

        # TODO: Replace with actual LLM call for realistic distortion
        # For MVP, simple rule-based distortion

        # If receiver strongly disagrees, tend to dismiss
        if stance_diff > 0.7 and receiver_confidence > 0.6:
            return f"[Filtered: Contradicts worldview] {content[:50]}...", True

        # If slight disagreement, soften
        if stance_diff > 0.3:
            return f"[Questionable] {content}", True

        return content, False

    def propagate_simple(
        self,
        message: Message,
        sender_id: str,
        recipient_ids: list[str],
        relationships: RelationshipMap,
    ) -> list[ReceivedMessage]:
        """Simple single-hop propagation.

        For MVP - no BFS, just direct recipients.

        Args:
            message: Message to propagate
            sender_id: Sender agent ID
            recipient_ids: List of recipient IDs
            relationships: Relationship map

        Returns:
            list[ReceivedMessage]: Received messages
        """
        received = []
        for rid in recipient_ids:
            edge = relationships.get(sender_id, rid)
            trust = max(0.0, edge.trust)

            import random
            if random.random() > trust:
                continue

            received.append(ReceivedMessage(
                message=message,
                from_agent=sender_id,
                to_agent=rid,
                depth=1,
                distortion_applied=False,
                trust_at_reception=trust,
            ))

        return received


def create_message(
    content: str,
    sender_id: str,
    topic_id: str,
    original_stance: float = 0.0,
    receiver_id: str = "",
) -> Message:
    """Create a new message.

    Args:
        content: Message content
        sender_id: Sender ID
        topic_id: Topic ID
        original_stance: Sender's stance
        receiver_id: Optional specific receiver

    Returns:
        Message: Created message
    """
    return Message(
        content=content,
        sender_id=sender_id,
        receiver_id=receiver_id,
        topic_id=topic_id,
        original_stance=original_stance,
        timestamp=datetime.now(),
    )
