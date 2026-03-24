"""Vectorized message propagation engine using NumPy.

Replaces Python BFS loops with NumPy vectorized operations for
massive performance improvement when propagating messages through
social networks.

Key optimization: Instead of iterating through neighbors one by one,
compute propagation for all edges simultaneously using matrix operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from utopia.core.pydantic_models import PropagationBatch

if TYPE_CHECKING:
    from utopia.layer4_social.network import SocialNetwork


@dataclass
class PropagationResult:
    """Result of a propagation operation.

    Attributes:
        reached_agents: Set of agents who received the message
        propagation_paths: List of (sender, receiver) tuples showing path
        depth_reached: Maximum propagation depth achieved
        successful_propagations: Count of successful message passes
    """

    reached_agents: set[str]
    propagation_paths: list[tuple[str, str, int]]  # (sender, receiver, depth)
    depth_reached: int
    successful_propagations: int


class VectorizedPropagator:
    """High-performance message propagation using NumPy vectorization.

    Algorithm overview:
    1. Convert agent network to adjacency matrix
    2. Represent message state as binary vector
    3. Use matrix multiplication to compute propagation in parallel
    4. Apply trust-based filtering using element-wise operations

    Complexity: O(d * n²) where d = depth, n = agents
    (vs O(d * e) for BFS where e = edges, but with much lower constant factors)
    """

    def __init__(
        self,
        agent_ids: list[str],
        trust_matrix: np.ndarray,
        trust_threshold: float = 0.0,
    ):
        """Initialize vectorized propagator.

        Args:
            agent_ids: Ordered list of agent IDs
            trust_matrix: Trust adjacency matrix [n_agents, n_agents]
            trust_threshold: Minimum trust for propagation
        """
        self.agent_ids = agent_ids
        self.agent_index = {aid: i for i, aid in enumerate(agent_ids)}
        self.n_agents = len(agent_ids)

        # Create binary adjacency matrix (1 if trust > threshold, else 0)
        self.adjacency = (trust_matrix >= trust_threshold).astype(np.float64)

        # Store trust values for probability weighting
        self.trust_matrix = np.clip(trust_matrix, -1, 1)

        # Precompute normalized trust for propagation probability
        # Higher trust = higher probability of accepting/sharing
        trust_normalized = (self.trust_matrix + 1) / 2  # [-1,1] -> [0,1]
        self.propagation_prob = trust_normalized

    def propagate(
        self,
        source_agents: list[str],
        max_depth: int = 3,
        topic_stance: Optional[float] = None,
        stance_tolerance: float = 0.5,
    ) -> PropagationResult:
        """Propagate message from source agents through network.

        Args:
            source_agents: Agents who initially have the message
            max_depth: Maximum propagation depth
            topic_stance: Optional stance value for the message
            stance_tolerance: How different agent stances can be from message

        Returns:
            Propagation result with paths and statistics
        """
        # Initialize state: 1 if agent has message, 0 otherwise
        has_message = np.zeros(self.n_agents, dtype=np.float64)
        source_indices = [self.agent_index[a] for a in source_agents if a in self.agent_index]
        has_message[source_indices] = 1.0

        # Track propagation paths
        all_paths: list[tuple[str, str, int]] = []
        reached_at_depth: dict[int, set[int]] = {0: set(source_indices)}

        # Track which agents have received message (ever)
        ever_received = has_message.copy()

        current_depth = 0

        while current_depth < max_depth:
            # Find agents who currently have message and will propagate
            current_holders = has_message > 0

            if not np.any(current_holders):
                break

            # Compute potential receivers: adjacency * current_holders
            # This gives count of how many current holders each agent is connected to
            potential_receivers = self.adjacency.T @ current_holders.astype(np.float64)

            # Apply propagation probability
            # For each potential receiver, probability = 1 - product(1 - p_i)
            # where p_i = trust-based probability from each holder
            propagation_scores = np.zeros(self.n_agents)

            for i in range(self.n_agents):
                if current_holders[i]:
                    # This agent holds the message - compute its contribution
                    trust_to_others = self.propagation_prob[i, :]
                    contribution = trust_to_others * self.adjacency[i, :]
                    propagation_scores += contribution

            # Agents receive message if propagation score > random threshold
            # For deterministic simulation, use threshold
            will_receive = (propagation_scores > 0.5) & (ever_received == 0)

            # Record paths for newly reached agents
            new_receivers = np.where(will_receive)[0]
            for receiver in new_receivers:
                # Find which holders sent to this receiver
                for holder in np.where(current_holders)[0]:
                    if self.adjacency[holder, receiver] > 0:
                        all_paths.append(
                            (self.agent_ids[holder], self.agent_ids[receiver], current_depth + 1)
                        )

            # Update states
            has_message = will_receive.astype(np.float64)
            ever_received = np.clip(ever_received + has_message, 0, 1)

            if np.any(has_message):
                reached_at_depth[current_depth + 1] = set(new_receivers)
                current_depth += 1
            else:
                break

        # Build result
        reached_indices = np.where(ever_received > 0)[0]
        reached_agents = {self.agent_ids[i] for i in reached_indices}

        return PropagationResult(
            reached_agents=reached_agents,
            propagation_paths=all_paths,
            depth_reached=current_depth,
            successful_propagations=len(all_paths),
        )

    def propagate_batch(
        self,
        batches: list[PropagationBatch],
        max_depth: int = 3,
    ) -> list[PropagationResult]:
        """Propagate multiple message batches in parallel.

        Args:
            batches: List of propagation batches
            max_depth: Maximum propagation depth

        Returns:
            List of propagation results
        """
        return [self._propagate_batch_single(batch, max_depth) for batch in batches]

    def _propagate_batch_single(
        self,
        batch: PropagationBatch,
        max_depth: int,
    ) -> PropagationResult:
        """Propagate a single batch.

        Args:
            batch: Propagation batch
            max_depth: Maximum depth

        Returns:
            Propagation result
        """
        # Convert batch sender IDs to indices
        source_indices = []
        for sender_id in batch.sender_ids:
            if sender_id in self.agent_index:
                source_indices.append(self.agent_index[sender_id])

        if not source_indices:
            return PropagationResult(
                reached_agents=set(),
                propagation_paths=[],
                depth_reached=0,
                successful_propagations=0,
            )

        # Weight by trust levels
        trust_weights = np.array(batch.trust_levels)
        if len(trust_weights) > 0:
            avg_trust = np.mean(np.clip(trust_weights, -1, 1))
        else:
            avg_trust = 0.0

        # Run propagation with trust-weighted probabilities
        has_message = np.zeros(self.n_agents, dtype=np.float64)
        has_message[source_indices] = 1.0

        ever_received = has_message.copy()
        all_paths: list[tuple[str, str, int]] = []
        current_depth = 0

        # Adjust propagation probability by batch trust
        adjusted_prob = self.propagation_prob * (0.5 + 0.5 * (avg_trust + 1) / 2)

        while current_depth < max_depth:
            current_holders = has_message > 0

            if not np.any(current_holders):
                break

            # Vectorized propagation computation
            propagation_scores = np.zeros(self.n_agents)

            holder_indices = np.where(current_holders)[0]
            if len(holder_indices) > 0:
                # Sum contributions from all holders (vectorized)
                contributions = adjusted_prob[holder_indices, :].sum(axis=0)
                propagation_scores = contributions

            # Apply threshold with trust-weighted randomness simulation
            threshold = 0.3 + 0.4 * (1 - avg_trust)  # Higher trust = lower threshold
            will_receive = (propagation_scores > threshold) & (ever_received == 0)

            # Record paths
            new_receivers = np.where(will_receive)[0]
            for receiver in new_receivers:
                for holder in holder_indices:
                    if self.adjacency[holder, receiver] > 0:
                        all_paths.append(
                            (self.agent_ids[holder], self.agent_ids[receiver], current_depth + 1)
                        )

            # Update states
            has_message = will_receive.astype(np.float64)
            ever_received = np.clip(ever_received + has_message, 0, 1)

            if np.any(has_message):
                current_depth += 1
            else:
                break

        reached_indices = np.where(ever_received > 0)[0]
        reached_agents = {self.agent_ids[i] for i in reached_indices}

        return PropagationResult(
            reached_agents=reached_agents,
            propagation_paths=all_paths,
            depth_reached=current_depth,
            successful_propagations=len(all_paths),
        )

    def compute_reach_probability(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
    ) -> float:
        """Compute probability that message from source reaches target.

        Args:
            source: Source agent ID
            target: Target agent ID
            max_depth: Maximum propagation depth

        Returns:
            Reach probability in [0, 1]
        """
        if source not in self.agent_index or target not in self.agent_index:
            return 0.0

        source_idx = self.agent_index[source]
        target_idx = self.agent_index[target]

        # Use matrix power to compute multi-hop probabilities
        # P(reach in d hops) = (adjacency ^ d)[source, target]

        prob = 0.0
        current_prob = np.eye(self.n_agents)  # Start with identity

        for depth in range(1, max_depth + 1):
            # Multiply by adjacency (weighted by trust)
            current_prob = current_prob @ (self.adjacency * self.propagation_prob)

            # Add probability of reaching at this depth
            depth_prob = current_prob[source_idx, target_idx]

            # Probability of first reaching at this depth
            prob += depth_prob * (1 - prob)

            if prob >= 0.99:
                break

        return float(np.clip(prob, 0, 1))


class FastBFSPropagator:
    """Optimized BFS propagator for small networks or single queries.

    Uses traditional BFS but optimized with NumPy for neighbor lookups.
    More memory-efficient than full matrix approach for sparse networks.
    """

    def __init__(
        self,
        agent_ids: list[str],
        trust_matrix: np.ndarray,
        trust_threshold: float = 0.0,
    ):
        """Initialize fast BFS propagator.

        Args:
            agent_ids: Ordered list of agent IDs
            trust_matrix: Trust adjacency matrix
            trust_threshold: Minimum trust for propagation
        """
        self.agent_ids = agent_ids
        self.agent_index = {aid: i for i, aid in enumerate(agent_ids)}
        self.n_agents = len(agent_ids)

        # Store sparse representation: dict of neighbor lists
        self.neighbors: dict[int, list[tuple[int, float]]] = {}

        for i in range(self.n_agents):
            self.neighbors[i] = []
            for j in range(self.n_agents):
                if i != j and trust_matrix[i, j] >= trust_threshold:
                    self.neighbors[i].append((j, trust_matrix[i, j]))

    def propagate(
        self,
        source_agents: list[str],
        max_depth: int = 3,
    ) -> PropagationResult:
        """Propagate using optimized BFS.

        Args:
            source_agents: Initial message holders
            max_depth: Maximum depth

        Returns:
            Propagation result
        """
        # Initialize BFS
        queue: list[tuple[int, int]] = []  # (agent_idx, depth)
        visited = set()
        paths: list[tuple[str, str, int]] = []

        for agent_id in source_agents:
            if agent_id in self.agent_index:
                idx = self.agent_index[agent_id]
                queue.append((idx, 0))
                visited.add(idx)

        head = 0
        max_reached_depth = 0

        while head < len(queue):
            current_idx, depth = queue[head]
            head += 1

            if depth >= max_depth:
                continue

            max_reached_depth = max(max_reached_depth, depth)

            # Process neighbors
            for neighbor_idx, trust in self.neighbors.get(current_idx, []):
                # Probabilistic propagation based on trust
                prob = (trust + 1) / 2  # [-1, 1] -> [0, 1]

                if neighbor_idx not in visited and np.random.random() < prob:
                    visited.add(neighbor_idx)
                    queue.append((neighbor_idx, depth + 1))
                    paths.append(
                        (self.agent_ids[current_idx], self.agent_ids[neighbor_idx], depth + 1)
                    )

        reached_agents = {self.agent_ids[i] for i in visited}

        return PropagationResult(
            reached_agents=reached_agents,
            propagation_paths=paths,
            depth_reached=max_reached_depth,
            successful_propagations=len(paths),
        )


def create_propagator(
    agent_ids: list[str],
    trust_matrix: np.ndarray,
    trust_threshold: float = 0.0,
    use_vectorized: bool = True,
) -> VectorizedPropagator | FastBFSPropagator:
    """Factory function to create appropriate propagator.

    Args:
        agent_ids: Ordered list of agent IDs
        trust_matrix: Trust adjacency matrix
        trust_threshold: Minimum trust for propagation
        use_vectorized: Use vectorized (True) or BFS (False) approach

    Returns:
        Configured propagator instance
    """
    if use_vectorized and len(agent_ids) >= 20:
        return VectorizedPropagator(agent_ids, trust_matrix, trust_threshold)
    else:
        return FastBFSPropagator(agent_ids, trust_matrix, trust_threshold)
