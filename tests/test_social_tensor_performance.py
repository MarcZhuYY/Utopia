"""Performance tests for SocialTensorGraph.

Tests the NumPy tensor-based implementation meets performance targets:
- 500 agents × 100 trust updates < 0.1 seconds
"""

import time

import numpy as np
import pytest

from utopia.layer4_social.social_network_tensor import SocialTensorGraph


class TestSocialTensorGraphPerformance:
    """Performance tests for SocialTensorGraph operations."""

    @pytest.fixture
    def large_graph(self):
        """Create a graph with 500 agents and 10 topics."""
        N = 500
        K = 10
        agent_ids = [f"A{i:03d}" for i in range(N)]
        topic_ids = [f"topic_{i}" for i in range(K)]

        graph = SocialTensorGraph(agent_ids, topic_ids)

        # 随机初始化信任矩阵
        graph.T = np.random.uniform(-0.5, 0.5, (N, N)).astype(np.float32)
        np.fill_diagonal(graph.T, 1.0)  # 自己对自己信任为 1

        # 随机初始化立场矩阵
        graph.S = np.random.uniform(-1, 1, (N, K)).astype(np.float32)

        return graph, N, K

    def test_trust_update_performance(self, large_graph):
        """性能测试：500节点，100次trust更新 < 0.1秒。

        Target: 100 updates < 0.1s (single update < 1ms)
        """
        graph, N, K = large_graph

        # 创建随机交互掩码（10% 交互概率）
        interaction_mask = np.random.choice(
            [0, 1], size=(N, N), p=[0.9, 0.1]
        ).astype(np.float32)

        # 预热（排除初始化开销）
        _ = graph.update_trust_matrix(
            topic_idx=0,
            interaction_mask=interaction_mask,
            learning_rate=0.1,
        )

        # 正式计时测试
        start = time.perf_counter()
        for _ in range(100):
            _ = graph.update_trust_matrix(
                topic_idx=0,
                interaction_mask=interaction_mask,
                learning_rate=0.1,
            )
        elapsed = time.perf_counter() - start

        print(f"\n100 trust updates took {elapsed:.4f}s")
        print(f"Average per update: {elapsed / 100 * 1000:.3f}ms")

        # 断言：100 次更新必须在 0.1 秒内完成
        assert elapsed < 0.1, f"100 updates took {elapsed:.3f}s, expected < 0.1s"

    def test_trust_update_single_call_performance(self, large_graph):
        """测试单次 trust 更新性能 < 2ms。"""
        graph, N, K = large_graph

        interaction_mask = np.random.choice(
            [0, 1], size=(N, N), p=[0.9, 0.1]
        ).astype(np.float32)

        # 单次调用计时
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = graph.update_trust_matrix(
                topic_idx=0,
                interaction_mask=interaction_mask,
                learning_rate=0.1,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        print(f"\nSingle trust update: {avg_time * 1000:.3f}ms")

        # 单次更新应小于 2ms (放宽阈值以适应不同硬件)
        assert avg_time < 0.002, f"Single update took {avg_time * 1000:.3f}ms, expected < 2ms"

    def test_matrix_propagation_performance(self, large_graph):
        """测试矩阵传播性能。"""
        graph, N, K = large_graph

        sender_idx = 0  # 第一个 agent

        # 预热
        _ = graph.propagate_message_matrix(
            sender_idx=sender_idx,
            propagation_threshold=0.1,
            max_hops=3,
        )

        # 计时 100 次传播
        start = time.perf_counter()
        for _ in range(100):
            probs = graph.propagate_message_matrix(
                sender_idx=sender_idx,
                propagation_threshold=0.1,
                max_hops=3,
            )
        elapsed = time.perf_counter() - start

        print(f"\n100 matrix propagations took {elapsed:.4f}s")
        print(f"Average per propagation: {elapsed / 100 * 1000:.3f}ms")

        # 100 次传播应在 0.1 秒内完成
        assert elapsed < 0.1, f"100 propagations took {elapsed:.3f}s"

    def test_affinity_computation_performance(self, large_graph):
        """测试亲和力矩阵计算性能。"""
        graph, N, K = large_graph

        # 计时 100 次亲和力计算
        start = time.perf_counter()
        for _ in range(100):
            affinity = graph.compute_affinity_matrix(topic_idx=0)
        elapsed = time.perf_counter() - start

        print(f"\n100 affinity computations took {elapsed:.4f}s")

        # 100 次计算应在 0.1 秒内完成 (放宽阈值)
        assert elapsed < 0.1

    def test_memory_efficiency(self, large_graph):
        """测试内存占用。"""
        graph, N, K = large_graph

        # 计算矩阵内存占用
        trust_mem = graph.T.nbytes
        stance_mem = graph.S.nbytes
        influence_mem = graph.W.nbytes
        familiarity_mem = graph.F.nbytes

        total_mem = trust_mem + stance_mem + influence_mem + familiarity_mem

        print(f"\nMemory usage:")
        print(f"  Trust matrix (T): {trust_mem / 1024:.1f} KB")
        print(f"  Stance matrix (S): {stance_mem / 1024:.1f} KB")
        print(f"  Influence matrix (W): {influence_mem / 1024:.1f} KB")
        print(f"  Familiarity matrix (F): {familiarity_mem / 1024:.1f} KB")
        print(f"  Total: {total_mem / 1024:.1f} KB")

        # 500 节点应在 10MB 以内
        assert total_mem < 10 * 1024 * 1024, f"Memory {total_mem / 1024 / 1024:.1f}MB > 10MB"


class TestSocialTensorGraphCorrectness:
    """Correctness tests for SocialTensorGraph operations."""

    @pytest.fixture
    def small_graph(self):
        """Create a small graph for correctness testing."""
        agent_ids = ["A", "B", "C", "D"]
        topic_ids = ["t1", "t2"]

        graph = SocialTensorGraph(agent_ids, topic_ids)

        # 设置初始立场
        # A 和 B 相似，C 和 D 相似，两组对立
        # t1: A=0.8, B=0.7, C=-0.7, D=-0.8
        graph.S[0, 0] = 0.8   # A on t1
        graph.S[1, 0] = 0.7   # B on t1
        graph.S[2, 0] = -0.7  # C on t1
        graph.S[3, 0] = -0.8  # D on t1

        # 初始信任矩阵
        graph.T = np.array([
            [1.0, 0.1, -0.1, -0.2],   # A
            [0.1, 1.0, -0.2, -0.1],   # B
            [-0.1, -0.2, 1.0, 0.2],   # C
            [-0.2, -0.1, 0.2, 1.0],   # D
        ], dtype=np.float32)

        return graph

    def test_homophily_update_increases_similar_trust(self, small_graph):
        """测试同质性更新增加相似立场 agent 之间的信任。"""
        graph = small_graph

        # 所有 agent 之间都有交互
        interaction_mask = np.ones((4, 4), dtype=np.float32)

        # 记录更新前的信任
        trust_before = graph.T[0, 1]  # A 对 B 的信任

        # 执行更新（高学习率以便观察变化）
        graph.update_trust_matrix(
            topic_idx=0,
            interaction_mask=interaction_mask,
            learning_rate=0.5,
        )

        # A 和 B 立场相似，信任应该增加
        trust_after = graph.T[0, 1]
        assert trust_after > trust_before, f"Trust should increase: {trust_before} -> {trust_after}"

    def test_homophily_update_decreases_dissimilar_trust(self, small_graph):
        """测试同质性更新减少不同立场 agent 之间的信任。"""
        graph = small_graph

        interaction_mask = np.ones((4, 4), dtype=np.float32)

        # 记录更新前的信任
        trust_before = graph.T[0, 2]  # A 对 C 的信任（A支持，C反对）

        # 执行更新
        graph.update_trust_matrix(
            topic_idx=0,
            interaction_mask=interaction_mask,
            learning_rate=0.5,
        )

        # A 和 C 立场对立，信任应该减少
        trust_after = graph.T[0, 2]
        assert trust_after < trust_before, f"Trust should decrease: {trust_before} -> {trust_after}"

    def test_affinity_matrix_computation(self, small_graph):
        """测试亲和力矩阵计算。"""
        graph = small_graph

        affinity = graph.compute_affinity_matrix(topic_idx=0)

        # A 和 B 立场接近 (0.8 vs 0.7)，亲和力应接近 1
        affinity_AB = affinity[0, 1]
        assert affinity_AB > 0.9, f"A-B affinity should be high: {affinity_AB}"

        # A 和 C 立场对立 (0.8 vs -0.7)，diff=1.5, affinity=1-1.5/2=0.25
        affinity_AC = affinity[0, 2]
        assert affinity_AC < 0.3, f"A-C affinity should be low: {affinity_AC}"

    def test_propagation_probabilities(self, small_graph):
        """测试传播概率计算。"""
        graph = small_graph

        # 从 A 开始传播
        probs = graph.propagate_message_matrix(
            sender_idx=0,  # A
            propagation_threshold=0.0,  # 不过滤，看所有概率
            decay_factor=1.0,  # 无衰减，简化验证
            max_hops=1,
        )

        # A 自己不接收
        assert probs[0] == 0.0

        # B 的信任度是 0.1，接收概率应为 0.1
        assert np.isclose(probs[1], 0.1, atol=0.01)

        # C 的信任度是 -0.1，但概率应为绝对值或 clip 到 0
        # 根据实现，负信任应该被处理
        assert probs[2] >= 0.0

    def test_trust_clipping(self, small_graph):
        """测试信任值被正确截断到 [-1, 1]。"""
        graph = small_graph

        # 设置极端立场差异以产生大的信任变化
        graph.S[:, 0] = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)

        interaction_mask = np.ones((4, 4), dtype=np.float32)

        # 高学习率应该导致大的更新
        graph.update_trust_matrix(
            topic_idx=0,
            interaction_mask=interaction_mask,
            learning_rate=1.0,
        )

        # 所有信任值应在 [-1, 1] 范围内
        assert np.all(graph.T >= -1.0)
        assert np.all(graph.T <= 1.0)

        # 对角线保持为 1
        assert np.all(np.diag(graph.T) == 1.0)


class TestSocialTensorGraphScaling:
    """Scaling tests for different graph sizes."""

    @pytest.mark.parametrize("N", [100, 200, 500, 1000])
    def test_scaling_trust_update(self, N):
        """测试不同规模下的 trust 更新性能。"""
        K = 10
        agent_ids = [f"A{i:03d}" for i in range(N)]
        topic_ids = [f"topic_{i}" for i in range(K)]

        graph = SocialTensorGraph(agent_ids, topic_ids)
        graph.T = np.random.uniform(-0.5, 0.5, (N, N)).astype(np.float32)
        np.fill_diagonal(graph.T, 1.0)
        graph.S = np.random.uniform(-1, 1, (N, K)).astype(np.float32)

        interaction_mask = np.random.choice(
            [0, 1], size=(N, N), p=[0.9, 0.1]
        ).astype(np.float32)

        # 计时 10 次更新
        start = time.perf_counter()
        for _ in range(10):
            _ = graph.update_trust_matrix(
                topic_idx=0,
                interaction_mask=interaction_mask,
                learning_rate=0.1,
            )
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 10
        print(f"\nN={N}: 10 updates took {elapsed:.4f}s, avg={avg_time*1000:.3f}ms")

        # 不同规模下的性能预期（宽松断言）
        if N <= 100:
            assert avg_time < 0.001  # < 1ms
        elif N <= 500:
            assert avg_time < 0.01   # < 10ms
        else:
            assert avg_time < 0.05   # < 50ms


class TestSocialTensorGraphComparison:
    """Compare with old dict-based implementation."""

    def test_numpy_vs_dict_memory(self):
        """比较 NumPy 矩阵 vs dict 的内存占用。"""
        N = 500

        # NumPy 矩阵内存
        trust_matrix = np.zeros((N, N), dtype=np.float32)
        numpy_mem = trust_matrix.nbytes

        # 估算 dict 内存（每个 RelationshipEdge 约 72 bytes + dict 开销）
        # dict: {agent_id: {other_id: RelationshipEdge}}
        # 假设每个 edge 72 bytes，每个 dict entry 约 72 bytes
        edges_count = N * N
        estimated_dict_mem = edges_count * (72 + 72)  # 粗略估计

        print(f"\nMemory comparison for {N} agents:")
        print(f"  NumPy matrix: {numpy_mem / 1024:.1f} KB")
        print(f"  Estimated dict: {estimated_dict_mem / 1024:.1f} KB")
        print(f"  Savings: {estimated_dict_mem / numpy_mem:.1f}x")

        # NumPy 应该节省至少 30 倍内存
        assert numpy_mem < estimated_dict_mem / 30

    def test_update_correctness_vs_homophily_engine(self):
        """验证 tensor update 结果与 HomophilyEngine 一致。"""
        from utopia.layer4_social.homophily import compute_affinity_matrix

        # 创建小规模图
        agent_ids = ["A", "B", "C"]
        topic_ids = ["t1"]

        graph = SocialTensorGraph(agent_ids, topic_ids)
        graph.S = np.array([[0.8], [0.6], [-0.5]], dtype=np.float32)

        # 使用独立函数计算亲和力 (传入 ndarray)
        stance_vector = graph.S[:, 0]  # shape: (3,)
        affinity_ref = compute_affinity_matrix(stance_vector)

        # 使用 SocialTensorGraph 计算
        affinity_tensor = graph.compute_affinity_matrix(topic_idx=0)

        # 验证结果一致
        assert np.allclose(affinity_ref, affinity_tensor, atol=0.001), \
            f"Affinity mismatch: {affinity_ref} vs {affinity_tensor}"
