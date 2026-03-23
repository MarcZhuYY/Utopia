"""Social Network Tensor Implementation - High-Performance NumPy-based L4 Layer.

This module provides SocialTensorGraph: a dense matrix representation of the social
network for极致性能. It replaces the inefficient dict[str, dict[str, RelationshipEdge]]
and NetworkX BFS with pure NumPy tensor operations.

Core Design:
- Pre-allocated float32 matrices for cache efficiency
- Vectorized operations using NumPy broadcasting
- Matrix multiplication for information propagation (替代 BFS)
- bidirectional index mapping: agent_id <-> matrix index

Performance Target:
- 500 agents × 100 trust updates < 0.1 seconds (single update < 1ms)
- 100-1000x speedup over Python dict + NetworkX implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


class SocialTensorGraph:
    """社会网络的 NumPy 张量化表示。

    使用密集矩阵替代 dict + NetworkX，实现向量化计算。

    Attributes:
        N: Agent 数量
        K: 议题数量
        agent_id_to_idx: Agent ID 到矩阵索引的映射
        idx_to_agent_id: 矩阵索引到 Agent ID 的映射
        topic_id_to_idx: 议题 ID 到矩阵索引的映射
        T: 信任矩阵 (N, N), float32, T[i,j] = i 对 j 的信任度 [-1, 1]
        W: 影响力矩阵 (N, N), float32, W[i,j] = i 对 j 的影响力 [0, 1]
        S: 立场矩阵 (N, K), float32, S[i,k] = i 对议题 k 的立场 [-1, 1]
        F: 熟悉度矩阵 (N, N), float32, F[i,j] = i 对 j 的熟悉度 [0, 1]
    """

    def __init__(
        self,
        agent_ids: list[str],
        topic_ids: list[str],
        dtype: np.dtype = np.float32,
    ):
        """初始化张量图。

        Args:
            agent_ids: Agent ID 列表
            topic_ids: 议题 ID 列表
            dtype: 矩阵数据类型，默认 float32
        """
        self.N = len(agent_ids)
        self.K = len(topic_ids)
        self.dtype = dtype

        # 创建双向索引映射
        self.agent_id_to_idx: dict[str, int] = {
            aid: i for i, aid in enumerate(agent_ids)
        }
        self.idx_to_agent_id: dict[int, str] = {
            i: aid for i, aid in enumerate(agent_ids)
        }
        self.topic_id_to_idx: dict[str, int] = {
            tid: i for i, tid in enumerate(topic_ids)
        }
        self.idx_to_topic_id: dict[int, str] = {
            i: tid for i, tid in enumerate(topic_ids)
        }

        # 预分配矩阵（内存一次性分配，避免 tick 内 GC）
        self.T: np.ndarray = np.zeros((self.N, self.N), dtype=dtype)  # Trust
        self.W: np.ndarray = np.zeros((self.N, self.N), dtype=dtype)  # Influence
        self.S: np.ndarray = np.zeros((self.N, self.K), dtype=dtype)  # Stance
        self.F: np.ndarray = np.zeros((self.N, self.N), dtype=dtype)  # Familiarity

        # 对角线初始化为特殊值（自己对自己）
        np.fill_diagonal(self.T, 1.0)  # 自己对自己完全信任
        np.fill_diagonal(self.W, 0.0)  # 自己对自己无影响力
        np.fill_diagonal(self.F, 1.0)  # 自己对自己完全熟悉

    # ========================================================================
    # Sync 接口: Agent 对象 <-> 矩阵 双向同步
    # ========================================================================

    def sync_from_agents(
        self,
        agents: dict[str, Agent],
        sync_stance: bool = True,
        sync_trust: bool = True,
    ) -> None:
        """Tick 开始时：将 Agent 对象数据刷入矩阵。

        Args:
            agents: Agent ID -> Agent 对象的映射
            sync_stance: 是否同步立场数据
            sync_trust: 是否同步信任数据
        """
        for agent_id, agent in agents.items():
            if agent_id not in self.agent_id_to_idx:
                continue  # 跳过未注册的 agent

            idx = self.agent_id_to_idx[agent_id]

            # 同步立场数据到 S 矩阵
            if sync_stance and hasattr(agent, 'beliefs') and agent.beliefs:
                for topic_id, stance in agent.beliefs.stances.items():
                    if topic_id not in self.topic_id_to_idx:
                        continue
                    topic_idx = self.topic_id_to_idx[topic_id]
                    self.S[idx, topic_idx] = stance.position

            # 同步信任数据到 T 矩阵
            if sync_trust:
                self._sync_agent_trust(idx, agent, agents)

    def _sync_agent_trust(
        self,
        idx: int,
        agent: Agent,
        all_agents: dict[str, Agent],
    ) -> None:
        """同步单个 agent 的信任关系到 T 矩阵。"""
        for other_id, other_agent in all_agents.items():
            if other_id not in self.agent_id_to_idx:
                continue

            other_idx = self.agent_id_to_idx[other_id]

            # 获取信任值
            if hasattr(agent, 'get_trust'):
                trust_value = agent.get_trust(other_id)
            elif hasattr(agent, '_relationship_map') and agent._relationship_map:
                edge = agent._relationship_map.get(agent.id, other_id)
                trust_value = edge.trust
            else:
                trust_value = 0.0

            self.T[idx, other_idx] = trust_value

    def sync_to_agents(
        self,
        agents: dict[str, Agent],
        sync_stance: bool = True,
        sync_trust: bool = True,
    ) -> None:
        """Tick 结束时：将矩阵计算结果刷回 Agent 对象。

        Args:
            agents: Agent ID -> Agent 对象的映射
            sync_stance: 是否同步立场数据
            sync_trust: 是否同步信任数据
        """
        for agent_id, agent in agents.items():
            if agent_id not in self.agent_id_to_idx:
                continue

            idx = self.agent_id_to_idx[agent_id]

            # 同步立场数据回 Agent
            if sync_stance:
                for topic_idx, topic_id in self.idx_to_topic_id.items():
                    stance_value = self.S[idx, topic_idx]
                    if hasattr(agent.beliefs, 'set_stance'):
                        agent.beliefs.set_stance(topic_id, stance_value)

            # 同步信任数据回 Agent
            if sync_trust:
                self._sync_trust_to_agent(idx, agent)

    def _sync_trust_to_agent(self, idx: int, agent: Agent) -> None:
        """将 T 矩阵的信任数据刷回单个 Agent。"""
        for other_idx, other_id in self.idx_to_agent_id.items():
            trust_value = self.T[idx, other_idx]

            # 更新到 relationship_map
            if hasattr(agent, '_relationship_map') and agent._relationship_map:
                from utopia.layer4_social.relationships import RelationshipEdge
                edge = agent._relationship_map.get(agent.id, other_id)
                edge.trust = float(trust_value)
                agent._relationship_map.set(agent.id, other_id, edge)

    # ========================================================================
    # 向量化 Homophily 信任更新
    # ========================================================================

    def update_trust_matrix(
        self,
        topic_idx: int,
        interaction_mask: np.ndarray,
        learning_rate: float = 0.1,
    ) -> np.ndarray:
        """向量化信任矩阵更新（homophily 机制）。

        使用纯 NumPy 广播机制，绝对禁止 for 循环。

        Formula:
            S_k = S[:, topic_idx]                              # (N,)
            Affinity = 1.0 - (|S_k[:, None] - S_k[None, :]| / 2.0)  # (N, N)
            Delta_T = (Affinity - 0.5) * mask * learning_rate  # (N, N)
            T_new = clip(T + Delta_T, -1.0, 1.0)

        Args:
            topic_idx: 议题索引
            interaction_mask: (N, N) 交互掩码，标记本轮哪些 Agent 对发生了交互
            learning_rate: 信任更新学习率

        Returns:
            np.ndarray: 更新后的信任矩阵 T
        """
        if topic_idx < 0 or topic_idx >= self.K:
            raise ValueError(f"topic_idx {topic_idx} out of range [0, {self.K})")

        # 提取议题立场向量 S_k: shape (N,)
        S_k = self.S[:, topic_idx]

        # 广播计算立场差分矩阵: |S_i - S_j|
        # S_k[:, np.newaxis]: (N, 1), S_k[np.newaxis, :]: (1, N)
        # 相减得到 (N, N) 差分矩阵
        stance_diff = np.abs(S_k[:, np.newaxis] - S_k[np.newaxis, :])

        # 计算亲和力矩阵 Affinity: (N, N)
        # Affinity = 1 - |S_i - S_j| / 2
        affinity = 1.0 - (stance_diff / 2.0)

        # 计算信任增量: (N, N)
        # Delta_T = (Affinity - 0.5) * mask * lr
        delta_T = (affinity - 0.5) * interaction_mask * learning_rate

        # 应用更新并截断到 [-1, 1]
        self.T = np.clip(self.T + delta_T, -1.0, 1.0)

        # 保持对角线为 1（自己对自己完全信任）
        np.fill_diagonal(self.T, 1.0)

        return self.T

    def update_trust_matrix_by_topic_id(
        self,
        topic_id: str,
        interaction_mask: np.ndarray,
        learning_rate: float = 0.1,
    ) -> np.ndarray:
        """通过议题 ID 更新信任矩阵的便捷方法。"""
        if topic_id not in self.topic_id_to_idx:
            raise ValueError(f"Unknown topic_id: {topic_id}")
        topic_idx = self.topic_id_to_idx[topic_id]
        return self.update_trust_matrix(topic_idx, interaction_mask, learning_rate)

    # ========================================================================
    # 矩阵化信息传播（替代 BFS）
    # ========================================================================

    def propagate_message_matrix(
        self,
        sender_idx: int,
        propagation_threshold: float = 0.3,
        decay_factor: float = 0.7,
        max_hops: int = 3,
    ) -> np.ndarray:
        """矩阵化信息传播，返回各节点接收概率。

        使用矩阵乘法 V @ T 替代传统 BFS 队列遍历。

        Algorithm:
            V_0: 初始激活向量，仅在 sender_idx 处为 1
            for hop in range(max_hops):
                V_next = V @ T           # 矩阵乘法传播
                V_next *= decay^hop      # 应用衰减
                probs += V_next          # 累积概率
                V = V_next               # 下一跳

        Args:
            sender_idx: 发送者索引
            propagation_threshold: 传播概率阈值，低于此值的节点不接收
            decay_factor: 每跳衰减因子
            max_hops: 最大传播跳数

        Returns:
            np.ndarray: (N,) 各节点接收概率
        """
        # 初始激活向量 V_0: shape (N,)
        V = np.zeros(self.N, dtype=self.dtype)
        V[sender_idx] = 1.0

        # 累积接收概率
        received_probs = np.zeros(self.N, dtype=self.dtype)

        for hop in range(max_hops):
            # 矩阵乘法传播: V_next = V @ T
            # V: (N,), T: (N, N), result: (N,)
            V_next = V @ self.T

            # 应用衰减
            decay = decay_factor ** hop
            V_next *= decay

            # 累积概率
            received_probs += V_next

            # 准备下一跳
            V = V_next

        # 应用阈值过滤
        received_probs[received_probs < propagation_threshold] = 0.0

        # 发送者不接收自己的消息
        received_probs[sender_idx] = 0.0

        return received_probs

    def get_receivers_by_probability(
        self,
        sender_idx: int,
        propagation_threshold: float = 0.3,
        **kwargs,
    ) -> list[tuple[str, float]]:
        """获取按概率排序的接收者列表。

        Args:
            sender_idx: 发送者索引
            propagation_threshold: 传播概率阈值
            **kwargs: 传递给 propagate_message_matrix 的参数

        Returns:
            list[tuple[str, float]]: [(receiver_id, probability), ...]
        """
        probs = self.propagate_message_matrix(
            sender_idx, propagation_threshold, **kwargs
        )

        # 获取非零概率的接收者
        receiver_indices = np.where(probs > 0)[0]

        # 构建 (agent_id, probability) 列表
        receivers = [
            (self.idx_to_agent_id[idx], float(probs[idx]))
            for idx in receiver_indices
        ]

        # 按概率降序排序
        receivers.sort(key=lambda x: x[1], reverse=True)

        return receivers

    # ========================================================================
    # 批量操作接口
    # ========================================================================

    def compute_affinity_matrix(self, topic_idx: int) -> np.ndarray:
        """计算指定议题的亲和力矩阵 Affinity。

        Formula: Affinity[i,j] = 1 - |S[i,k] - S[j,k]| / 2

        Args:
            topic_idx: 议题索引

        Returns:
            np.ndarray: (N, N) 亲和力矩阵
        """
        S_k = self.S[:, topic_idx]
        stance_diff = np.abs(S_k[:, np.newaxis] - S_k[np.newaxis, :])
        return 1.0 - (stance_diff / 2.0)

    def compute_polarization_index(self, topic_idx: int) -> float:
        """计算议题的极化指数。

        使用立场分布的标准差作为极化指标。

        Args:
            topic_idx: 议题索引

        Returns:
            float: 极化指数 [0, 1]，越高越极化
        """
        stances = self.S[:, topic_idx]
        # 标准差归一化到 [0, 1]
        std = np.std(stances)
        return float(std)  # 立场范围 [-1, 1]，std 最大约 1.0

    def get_echo_chambers(
        self,
        topic_idx: int,
        affinity_threshold: float = 0.7,
    ) -> list[list[str]]:
        """识别回声室（高亲和力子群）。

        Args:
            topic_idx: 议题索引
            affinity_threshold: 亲和力阈值，高于此值的认为在同一回声室

        Returns:
            list[list[str]]: 回声室列表，每个回声室是 Agent ID 列表
        """
        affinity = self.compute_affinity_matrix(topic_idx)

        # 构建邻接矩阵（亲和力高于阈值）
        adjacency = affinity >= affinity_threshold

        # 使用简单的连通分量检测（可用 scipy 优化）
        visited = np.zeros(self.N, dtype=bool)
        chambers = []

        for i in range(self.N):
            if visited[i]:
                continue

            # BFS 找到连通分量
            component = []
            queue = [i]
            visited[i] = True

            while queue:
                node = queue.pop(0)
                component.append(node)

                # 找到所有未访问的邻居
                neighbors = np.where(adjacency[node] & ~visited)[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)

            if len(component) > 1:  # 忽略孤立节点
                chamber = [self.idx_to_agent_id[idx] for idx in component]
                chambers.append(chamber)

        return chambers

    # ========================================================================
    # 矩阵访问接口
    # ========================================================================

    def get_trust(self, agent_a_id: str, agent_b_id: str) -> float:
        """获取两个 Agent 之间的信任值。"""
        idx_a = self.agent_id_to_idx[agent_a_id]
        idx_b = self.agent_id_to_idx[agent_b_id]
        return float(self.T[idx_a, idx_b])

    def set_trust(self, agent_a_id: str, agent_b_id: str, value: float) -> None:
        """设置两个 Agent 之间的信任值。"""
        idx_a = self.agent_id_to_idx[agent_a_id]
        idx_b = self.agent_id_to_idx[agent_b_id]
        self.T[idx_a, idx_b] = np.clip(value, -1.0, 1.0)

    def get_stance(self, agent_id: str, topic_id: str) -> float:
        """获取 Agent 对议题的立场。"""
        idx_a = self.agent_id_to_idx[agent_id]
        idx_t = self.topic_id_to_idx[topic_id]
        return float(self.S[idx_a, idx_t])

    def set_stance(self, agent_id: str, topic_id: str, value: float) -> None:
        """设置 Agent 对议题的立场。"""
        idx_a = self.agent_id_to_idx[agent_id]
        idx_t = self.topic_id_to_idx[topic_id]
        self.S[idx_a, idx_t] = np.clip(value, -1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "N": self.N,
            "K": self.K,
            "agent_ids": list(self.agent_id_to_idx.keys()),
            "topic_ids": list(self.topic_id_to_idx.keys()),
            "T": self.T.tolist(),
            "W": self.W.tolist(),
            "S": self.S.tolist(),
            "F": self.F.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SocialTensorGraph":
        """从字典反序列化。"""
        graph = cls(
            agent_ids=data["agent_ids"],
            topic_ids=data["topic_ids"],
        )
        graph.T = np.array(data["T"], dtype=graph.dtype)
        graph.W = np.array(data["W"], dtype=graph.dtype)
        graph.S = np.array(data["S"], dtype=graph.dtype)
        graph.F = np.array(data["F"], dtype=graph.dtype)
        return graph
