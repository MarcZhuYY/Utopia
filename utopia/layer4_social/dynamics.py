"""Group dynamics detection.

Detects polarization, opinion leaders, and other group-level phenomena.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from utopia.layer3_cognition.beliefs import BayesianBeliefSystem

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


@dataclass
class PolarizationReport:
    """Report on group polarization.

    Attributes:
        polarized: Whether significant polarization exists
        score: Polarization score (0-1)
        mean: Mean stance position
        std: Standard deviation of stances
        bimodal_score: Score for bimodal distribution (0-1)
        clusters: Identified stance clusters
    """

    polarized: bool = False
    score: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    bimodal_score: float = 0.0
    clusters: list[list[str]] = None

    def __post_init__(self):
        if self.clusters is None:
            self.clusters = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "polarized": self.polarized,
            "score": self.score,
            "mean": self.mean,
            "std": self.std,
            "bimodal_score": self.bimodal_score,
            "clusters": self.clusters,
        }


@dataclass
class OpinionLeader:
    """Represents an opinion leader.

    Attributes:
        agent_id: Agent ID
        score: Leadership score
        topics: Topics they lead on
    """

    agent_id: str
    score: float
    topics: list[str]


class GroupDynamicsDetector:
    """Detects group-level dynamics and patterns.

    Capabilities:
    - Polarization detection
    - Opinion leader identification
    - Trend analysis
    """

    POLARIZATION_THRESHOLD = 0.4

    def detect_polarization(
        self,
        agents: list[Agent],
        topic_id: str,
    ) -> PolarizationReport:
        """Detect polarization on a topic.

        Args:
            agents: List of agents
            topic_id: Topic to analyze

        Returns:
            PolarizationReport: Polarization analysis
        """
        stances = []
        agent_ids = []

        for agent in agents:
            stance = agent.get_stance(topic_id)
            if stance:
                stances.append(stance.position)
                agent_ids.append(agent.id)

        if len(stances) < 2:
            return PolarizationReport()

        stances = np.array(stances)
        mean = float(np.mean(stances))
        std = float(np.std(stances))

        # Bimodal detection
        bimodal_score = self._detect_bimodal(stances)

        # Combined score
        score = (std * 0.6) + (bimodal_score * 0.4)

        polarized = score > self.POLARIZATION_THRESHOLD

        # Cluster agents by stance
        clusters = self._cluster_agents(stances, agent_ids)

        return PolarizationReport(
            polarized=polarized,
            score=score,
            mean=mean,
            std=std,
            bimodal_score=bimodal_score,
            clusters=clusters,
        )

    def _detect_bimodal(self, stances: np.ndarray) -> float:
        """Detect if distribution is bimodal.

        Uses simple gap detection.

        Args:
            stances: Array of stance positions

        Returns:
            float: Bimodal score (0-1)
        """
        if len(stances) < 4:
            return 0.0

        # Sort stances
        sorted_stances = np.sort(stances)

        # Find gaps
        diffs = np.diff(sorted_stances)
        if len(diffs) == 0:
            return 0.0

        max_gap = np.max(diffs)
        mean_gap = np.mean(diffs)

        # Large gap relative to spread indicates bimodality
        spread = np.max(sorted_stances) - np.min(sorted_stances)
        if spread < 0.1:
            return 0.0

        gap_ratio = max_gap / spread

        return min(1.0, gap_ratio * 2)

    def _cluster_agents(
        self,
        stances: np.ndarray,
        agent_ids: list[str],
    ) -> list[list[str]]:
        """Cluster agents by stance position.

        Args:
            stances: Array of stances
            agent_ids: Corresponding agent IDs

        Returns:
            list[list[str]]: Clusters of agent IDs
        """
        if len(stances) < 2:
            return [agent_ids]

        # Simple k-means with k=2 for MVP
        try:
            sorted_indices = np.argsort(stances)
            mid = len(stances) // 2

            cluster1 = [agent_ids[i] for i in sorted_indices[:mid]]
            cluster2 = [agent_ids[i] for i in sorted_indices[mid:]]

            clusters = []
            if cluster1:
                clusters.append(cluster1)
            if cluster2:
                clusters.append(cluster2)
            return clusters
        except Exception:
            return [agent_ids]

    def detect_opinion_leaders(
        self,
        agents: list[Agent],
        topic_id: str,
        top_n: int = 5,
    ) -> list[OpinionLeader]:
        """Identify opinion leaders on a topic.

        Opinion leaders have:
        - High influence
        - High confidence
        - Topic expertise

        Args:
            agents: List of agents
            topic_id: Topic to analyze
            top_n: Number of leaders to return

        Returns:
            list[OpinionLeader]: Top opinion leaders
        """
        scored = []

        for agent in agents:
            stance = agent.get_stance(topic_id)
            if not stance:
                continue

            # Influence score
            influence = agent.persona.influence_base

            # Confidence score
            confidence = stance.confidence

            # Expertise score
            expertise = 1.0 if topic_id in agent.persona.expertise else 0.3

            # Combined score
            score = influence * 0.4 + confidence * 0.3 + expertise * 0.3

            scored.append(OpinionLeader(
                agent_id=agent.id,
                score=score,
                topics=[topic_id],
            ))

        # Sort and return top N
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]

    def compute_group_sentiment(
        self,
        agents: list[Agent],
        topic_id: str,
    ) -> dict[str, float]:
        """Compute aggregate sentiment on a topic.

        Args:
            agents: List of agents
            topic_id: Topic to analyze

        Returns:
            dict: Sentiment metrics
        """
        stances = []
        confidences = []

        for agent in agents:
            stance = agent.get_stance(topic_id)
            if stance:
                stances.append(stance.position)
                confidences.append(stance.confidence)

        if not stances:
            return {"mean": 0.0, "std": 0.0, "positive_ratio": 0.5, "confidence_weighted_mean": 0.0}

        stances = np.array(stances)
        confidences = np.array(confidences)

        mean = float(np.mean(stances))
        std = float(np.std(stances))
        positive_ratio = float(np.mean(stances > 0))
        confidence_weighted_mean = float(np.average(stances, weights=confidences))

        return {
            "mean": mean,
            "std": std,
            "positive_ratio": positive_ratio,
            "confidence_weighted_mean": confidence_weighted_mean,
        }

    def detect_trend(
        self,
        history: list[dict[str, float]],
        topic_id: str,
    ) -> dict[str, Any]:
        """Detect trend in stance history.

        Args:
            history: List of sentiment dicts over time
            topic_id: Topic ID

        Returns:
            dict: Trend analysis
        """
        if len(history) < 2:
            return {"trend": "stable", "velocity": 0.0}

        means = [h.get("mean", 0.0) for h in history]

        # Simple velocity: change in mean
        velocity = means[-1] - means[0]

        # Determine trend direction
        if abs(velocity) < 0.05:
            trend = "stable"
        elif velocity > 0:
            trend = "positive"
        else:
            trend = "negative"

        return {
            "trend": trend,
            "velocity": velocity,
            "initial": means[0],
            "current": means[-1],
            "total_change": abs(velocity),
        }
