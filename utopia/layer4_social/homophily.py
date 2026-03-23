"""Homophily-based trust update and echo chamber effects.

This module implements homophily ("birds of a feather flock together") dynamics:
- Agents with similar stances develop stronger trust bonds
- Dissimilar stances erode trust over time
- Creates echo chamber effects where information circulates within clusters

Mathematical foundation:
Affinity = 1 - |S_a - S_b| / 2
Delta_Trust = (Affinity - 0.5) * Interaction_Quality * 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from utopia.core.pydantic_models import (
    BigFiveTraits,
    HomophilyUpdateInput,
    TrustUpdateResult,
)

if TYPE_CHECKING:
    from utopia.layer4_social.network import SocialNetwork

# SAFETY FIX: Epsilon for safe division
_EPSILON = 1e-10


def _sanitize_stance(
    value: float,
    default: float = 0.0,
    bounds: tuple[float, float] = (-1.0, 1.0)
) -> float:
    """Sanitize stance value to prevent NaN/Inf.

    SAFETY FIX: Prevents invalid stance values from corrupting homophily calculations.

    Args:
        value: Input stance value
        default: Default if NaN/Inf
        bounds: Valid range (min, max)

    Returns:
        Sanitized stance within bounds
    """
    if np.isnan(value) or np.isinf(value):
        return float(np.clip(default, bounds[0], bounds[1]))
    return float(np.clip(value, bounds[0], bounds[1]))


@dataclass
class HomophilyConfig:
    """Configuration for homophily dynamics.

    Attributes:
        update_rate: Base rate of trust change per interaction (0.1 = default)
        affinity_threshold: Minimum affinity for "friend" classification
        echo_chamber_threshold: Trust level for echo chamber membership
        max_trust_change: Maximum trust change per interaction
        decay_rate: Passive trust decay rate for non-interacting pairs
    """

    update_rate: float = 0.1
    affinity_threshold: float = 0.6  # Affinity > 0.6 = similar stances
    echo_chamber_threshold: float = 0.7  # Trust > 0.7 = echo chamber member
    max_trust_change: float = 0.15
    decay_rate: float = 0.01  # Passive decay per tick


class HomophilyEngine:
    """Engine for homophily-based trust dynamics.

    Implements the mathematical formula:
    - Affinity = 1 - |stance_a - stance_b| / 2
    - Trust delta = (affinity - 0.5) * quality * rate

    Creates natural clustering where agents with similar views
    form tightly connected communities (echo chambers).
    """

    def __init__(self, config: Optional[HomophilyConfig] = None):
        """Initialize homophily engine.

        Args:
            config: Homophily configuration (uses defaults if None)
        """
        self.config = config or HomophilyConfig()

    def compute_affinity(
        self,
        stance_a: float,
        stance_b: float,
    ) -> float:
        """Compute stance affinity between two agents.

        Affinity ranges from 0 (opposite stances: -1 vs 1)
        to 1 (identical stances).

        Formula: Affinity = 1 - |S_a - S_b| / 2

        Args:
            stance_a: Agent A's stance [-1, 1]
            stance_b: Agent B's stance [-1, 1]

        Returns:
            Affinity in [0, 1]
        """
        # SAFETY FIX: Sanitize inputs
        stance_a = _sanitize_stance(stance_a)
        stance_b = _sanitize_stance(stance_b)

        stance_diff = abs(stance_a - stance_b)
        return 1.0 - (stance_diff / 2.0)

    def compute_trust_delta(
        self,
        affinity: float,
        interaction_quality: float,
        trust_in_sender: float,
    ) -> float:
        """Compute trust change from homophily dynamics.

        Formula: Delta = (affinity - 0.5) * quality * rate * trust_factor

        - Positive affinity (>0.5) increases trust
        - Negative affinity (<0.5) decreases trust
        - Quality amplifies the effect
        - Existing trust moderates the change

        Args:
            affinity: Stance affinity [0, 1]
            interaction_quality: Interaction quality [0, 1]
            trust_in_sender: Current trust level [-1, 1]

        Returns:
            Trust delta (can be negative)
        """
        # SAFETY FIX: Sanitize inputs
        affinity = _sanitize_stance(affinity, default=0.5, bounds=(0.0, 1.0))
        interaction_quality = _sanitize_stance(interaction_quality, default=0.5, bounds=(0.0, 1.0))
        trust_in_sender = _sanitize_stance(trust_in_sender, default=0.0, bounds=(-1.0, 1.0))

        # Normalize trust to [0, 1] for calculation
        normalized_trust = (trust_in_sender + 1) / 2

        # Base delta from affinity
        # Affinity > 0.5 means similar stances -> trust increase
        # Affinity < 0.5 means dissimilar stances -> trust decrease
        affinity_effect = affinity - 0.5

        # Compute raw delta
        raw_delta = (
            affinity_effect *
            interaction_quality *
            self.config.update_rate *
            (0.5 + 0.5 * normalized_trust)  # Higher trust = more receptive
        )

        # Clamp to max change
        return float(np.clip(
            raw_delta,
            -self.config.max_trust_change,
            self.config.max_trust_change
        ))

    def update_trust(
        self,
        input_data: HomophilyUpdateInput,
    ) -> TrustUpdateResult:
        """Update trust based on homophily dynamics.

        Args:
            input_data: Validated update parameters

        Returns:
            Trust update result
        """
        # Compute affinity
        affinity = self.compute_affinity(
            input_data.stance_a,
            input_data.stance_b,
        )

        # Compute trust delta
        delta = self.compute_trust_delta(
            affinity,
            input_data.interaction_quality,
            input_data.current_trust,
        )

        # Apply delta
        new_trust = float(np.clip(
            input_data.current_trust + delta,
            -1.0,
            1.0
        ))

        # Generate reason
        if affinity > 0.7:
            reason = f"立场高度一致 (亲和度={affinity:.2f})，信任显著提升"
        elif affinity > 0.5:
            reason = f"立场相似 (亲和度={affinity:.2f})，信任稳步增加"
        elif affinity > 0.3:
            reason = f"立场分歧 (亲和度={affinity:.2f})，信任略有下降"
        else:
            reason = f"立场对立 (亲和度={affinity:.2f})，信任严重受损"

        return TrustUpdateResult(
            agent_a_id=input_data.agent_a_id,
            agent_b_id=input_data.agent_b_id,
            old_trust=input_data.current_trust,
            new_trust=new_trust,
            delta=new_trust - input_data.current_trust,
            affinity=affinity,
            reason=reason,
        )

    def batch_update_trust(
        self,
        interactions: list[HomophilyUpdateInput],
    ) -> list[TrustUpdateResult]:
        """Batch update trust for multiple interactions.

        Args:
            interactions: List of interaction inputs

        Returns:
            List of trust update results
        """
        return [self.update_trust(inp) for inp in interactions]

    def detect_echo_chambers(
        self,
        trust_matrix: dict[tuple[str, str], float],
        agent_stances: dict[str, float],
        min_size: int = 3,
    ) -> list[set[str]]:
        """Detect echo chambers in the network.

        Echo chambers are clusters where:
        1. All members have trust >= threshold with each other
        2. Members have similar stances (affinity >= threshold)

        Args:
            trust_matrix: Trust levels for agent pairs
            agent_stances: Current stance for each agent
            min_size: Minimum chamber size

        Returns:
            List of echo chambers (sets of agent IDs)
        """
        # Build adjacency graph based on trust threshold
        adjacency: dict[str, set[str]] = {}

        for (a, b), trust in trust_matrix.items():
            if trust >= self.config.echo_chamber_threshold:
                if a not in adjacency:
                    adjacency[a] = set()
                if b not in adjacency:
                    adjacency[b] = set()
                adjacency[a].add(b)
                adjacency[b].add(a)

        # Find cliques (groups where everyone trusts everyone)
        chambers: list[set[str]] = []
        visited: set[str] = set()

        def find_clique(start: str) -> set[str]:
            """Find maximal clique containing start node."""
            clique = {start}
            candidates = adjacency.get(start, set()).copy()

            while candidates:
                # Find candidate connected to all current clique members
                for candidate in list(candidates):
                    if all(
                        candidate in adjacency.get(member, set())
                        for member in clique
                    ):
                        clique.add(candidate)
                        candidates &= adjacency.get(candidate, set())
                    else:
                        candidates.discard(candidate)

            return clique

        for agent in adjacency:
            if agent not in visited:
                clique = find_clique(agent)
                visited.update(clique)

                # Check stance similarity within clique
                if len(clique) >= min_size:
                    stances = [agent_stances.get(a, 0) for a in clique]
                    stance_variance = np.var(stances)

                    # Low variance = similar stances = echo chamber
                    if stance_variance < 0.2:
                        chambers.append(clique)

        return chambers

    def compute_polarization_index(
        self,
        trust_matrix: dict[tuple[str, str], float],
        agent_stances: dict[str, float],
    ) -> float:
        """Compute network polarization index.

        High polarization means the network is divided into
        disconnected clusters with opposing views.

        Formula: Variance of average inter-cluster trust

        Args:
            trust_matrix: Trust levels for agent pairs
            agent_stances: Current stance for each agent

        Returns:
            Polarization index in [0, 1]
        """
        if not trust_matrix:
            return 0.0

        # Group agents by stance clusters
        sorted_agents = sorted(agent_stances.items(), key=lambda x: x[1])

        # Find natural stance clusters
        clusters: list[list[str]] = []
        current_cluster: list[str] = []
        last_stance = None

        for agent_id, stance in sorted_agents:
            if last_stance is None or abs(stance - last_stance) < 0.3:
                current_cluster.append(agent_id)
            else:
                clusters.append(current_cluster)
                current_cluster = [agent_id]
            last_stance = stance

        if current_cluster:
            clusters.append(current_cluster)

        if len(clusters) < 2:
            return 0.0

        # Compute inter-cluster trust levels
        inter_cluster_trusts = []

        for i, cluster_a in enumerate(clusters):
            for cluster_b in clusters[i + 1:]:
                trusts = []
                for a in cluster_a:
                    for b in cluster_b:
                        trust = trust_matrix.get((a, b), trust_matrix.get((b, a), 0))
                        trusts.append(trust)

                if trusts:
                    avg_trust = np.mean(trusts)
                    inter_cluster_trusts.append(avg_trust)

        if not inter_cluster_trusts:
            return 0.0

        # Polarization = variance of inter-cluster trust
        # High variance = some clusters trust each other, others don't
        trust_variance = np.var(inter_cluster_trusts)

        return float(np.clip(trust_variance, 0.0, 1.0))

    def passive_decay(
        self,
        trust_matrix: dict[tuple[str, str], float],
        last_interaction_tick: dict[tuple[str, str], int],
        current_tick: int,
        decay_window: int = 10,
    ) -> dict[tuple[str, str], float]:
        """Apply passive trust decay for non-interacting pairs.

        Trust naturally decays over time without interaction,
        accelerating echo chamber formation.

        Args:
            trust_matrix: Current trust levels
            last_interaction_tick: Last interaction time for each pair
            current_tick: Current simulation tick
            decay_window: Ticks before decay starts

        Returns:
            Updated trust matrix
        """
        updated = trust_matrix.copy()

        for pair, last_tick in last_interaction_tick.items():
            ticks_since = current_tick - last_tick

            if ticks_since > decay_window:
                # Apply decay
                current_trust = updated.get(pair, 0)
                decay_amount = self.config.decay_rate * (ticks_since - decay_window)

                # Decay toward 0 (neutral)
                if current_trust > 0:
                    updated[pair] = max(0.0, current_trust - decay_amount)
                else:
                    updated[pair] = min(0.0, current_trust + decay_amount)

        return updated


class EchoChamberAnalyzer:
    """Analyzer for echo chamber dynamics and information flow."""

    def __init__(self, homophily_engine: HomophilyEngine):
        """Initialize analyzer.

        Args:
            homophily_engine: Homophily engine instance
        """
        self.engine = homophily_engine

    def analyze_information_flow(
        self,
        trust_matrix: dict[tuple[str, str], float],
        message_propagation_path: list[tuple[str, str]],
    ) -> dict:
        """Analyze how information flows through the network.

        Args:
            trust_matrix: Trust levels between agents
            message_propagation_path: Sequence of message passes

        Returns:
            Flow analysis results
        """
        # Count intra-chamber vs inter-chamber propagation
        intra_chamber = 0
        inter_chamber = 0
        high_trust_passes = 0

        for sender, receiver in message_propagation_path:
            trust = trust_matrix.get(
                (sender, receiver),
                trust_matrix.get((receiver, sender), 0)
            )

            if trust > self.engine.config.echo_chamber_threshold:
                intra_chamber += 1
                high_trust_passes += 1
            elif trust > 0:
                inter_chamber += 1
                high_trust_passes += 1

        total = len(message_propagation_path)

        return {
            "total_propagations": total,
            "intra_chamber_ratio": intra_chamber / total if total > 0 else 0,
            "inter_chamber_ratio": inter_chamber / total if total > 0 else 0,
            "high_trust_ratio": high_trust_passes / total if total > 0 else 0,
            "echo_chamber_strength": intra_chamber / high_trust_passes
                if high_trust_passes > 0 else 0,
        }

    def identify_bridge_agents(
        self,
        trust_matrix: dict[tuple[str, str], float],
        agent_stances: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Identify bridge agents connecting different clusters.

        Bridge agents are crucial for information diversity as they
        maintain connections across echo chambers.

        Args:
            trust_matrix: Trust levels between agents
            agent_stances: Current stance for each agent

        Returns:
            List of (agent_id, bridge_score) tuples, sorted by score
        """
        # Build trust graph
        neighbors: dict[str, set[str]] = {}

        for (a, b), trust in trust_matrix.items():
            if trust > 0:  # Positive trust connection
                if a not in neighbors:
                    neighbors[a] = set()
                if b not in neighbors:
                    neighbors[b] = set()
                neighbors[a].add(b)
                neighbors[b].add(a)

        bridge_scores = []

        for agent_id in agent_stances:
            neighbor_list = list(neighbors.get(agent_id, []))

            if len(neighbor_list) < 2:
                continue

            # Bridge score = stance variance among neighbors
            # High variance means connecting different groups
            neighbor_stances = [
                agent_stances.get(n, 0) for n in neighbor_list
            ]
            stance_variance = np.var(neighbor_stances)

            # Also consider cross-chamber connections
            cross_chamber = sum(
                1 for n in neighbor_list
                if abs(agent_stances.get(n, 0) - agent_stances[agent_id]) > 0.5
            )

            bridge_score = stance_variance * (1 + cross_chamber)
            bridge_scores.append((agent_id, bridge_score))

        return sorted(bridge_scores, key=lambda x: x[1], reverse=True)

    def compute_opinion_diversity(
        self,
        agent_stances: dict[str, float],
        trust_matrix: dict[tuple[str, str], float],
    ) -> float:
        """Compute opinion diversity index.

        High diversity = many different opinions exist
        Low diversity = opinions are concentrated

        Args:
            agent_stances: Current stance for each agent
            trust_matrix: Trust levels (for weighting)

        Returns:
            Diversity index in [0, 1]
        """
        if not agent_stances:
            return 0.0

        # SAFETY FIX: Sanitize stance values
        stances = [
            _sanitize_stance(s) for s in agent_stances.values()
            if not (np.isnan(s) or np.isinf(s))
        ]

        if not stances:
            return 0.0

        # Use entropy-based diversity
        # Bin stances into categories
        bins = np.linspace(-1, 1, 5)  # -1 to 1 in 5 bins
        hist, _ = np.histogram(stances, bins=bins)

        # SAFETY FIX: Safe division with epsilon
        hist_sum = hist.sum()
        if hist_sum < _EPSILON:
            return 0.0

        # Compute normalized entropy
        probs = hist / hist_sum
        entropy = -sum(p * np.log(p + _EPSILON) for p in probs if p > 0)
        max_entropy = np.log(len(bins) - 1)

        # SAFETY FIX: Protect against zero max_entropy
        if max_entropy < _EPSILON:
            return 0.0

        return float(np.clip(entropy / max_entropy, 0.0, 1.0))


# =============================================================================
# Vectorized operations for performance
# =============================================================================


def compute_affinity_matrix(
    stance_vector: np.ndarray,
) -> np.ndarray:
    """Compute affinity matrix for all agent pairs.

    Args:
        stance_vector: Array of agent stances [n_agents]

    Returns:
        Affinity matrix [n_agents, n_agents]
    """
    # |S_i - S_j| for all pairs
    diff_matrix = np.abs(stance_vector[:, np.newaxis] - stance_vector[np.newaxis, :])

    # Affinity = 1 - |diff| / 2
    return 1.0 - diff_matrix / 2.0


def compute_trust_delta_matrix(
    affinity_matrix: np.ndarray,
    current_trust_matrix: np.ndarray,
    interaction_quality_matrix: np.ndarray,
    update_rate: float = 0.1,
) -> np.ndarray:
    """Compute trust deltas for all pairs simultaneously.

    Args:
        affinity_matrix: Affinity values [n_agents, n_agents]
        current_trust_matrix: Current trust levels [n_agents, n_agents]
        interaction_quality_matrix: Quality of interactions [n_agents, n_agents]
        update_rate: Base update rate

    Returns:
        Trust delta matrix [n_agents, n_agents]
    """
    # Normalize trust to [0, 1]
    normalized_trust = (current_trust_matrix + 1) / 2

    # Affinity effect (affinity - 0.5)
    affinity_effect = affinity_matrix - 0.5

    # Compute deltas
    deltas = (
        affinity_effect *
        interaction_quality_matrix *
        update_rate *
        (0.5 + 0.5 * normalized_trust)
    )

    # Clamp to reasonable range
    return np.clip(deltas, -0.15, 0.15)
