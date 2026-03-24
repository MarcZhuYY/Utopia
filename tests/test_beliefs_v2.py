"""Tests for Bayesian belief system (beliefs_v2.py)."""

import numpy as np
import pytest

from utopia.core.pydantic_models import BigFiveTraits, StanceState
from utopia.layer3_cognition.beliefs import (
    BayesianBeliefSystem,
    BayesianBeliefDelta,
    _sanitize_float,
    compute_belief_distance,
)


class TestBayesianBeliefSystem:
    """Test Bayesian belief update with mathematical rigor."""

    def test_bayesian_update_formula(self):
        """Test core Bayesian update formula:
        Delta = (S_m - S_i) * T_ij * I_m
        S_new = S_i + Delta * (1 - C_i) * O
        """
        system = BayesianBeliefSystem()

        # Initialize stance
        system.initialize_stance("topic1", position=0.0, confidence=0.5)

        # Update parameters
        message_stance = 0.8
        intensity = 0.5
        trust = 0.6
        openness = system.traits.openness  # default 0.5

        # Expected calculation
        old_position = 0.0
        old_confidence = 0.5
        expected_delta = (message_stance - old_position) * trust * intensity
        expected_delta_stance = expected_delta * (1 - old_confidence) * openness
        expected_new_position = old_position + expected_delta_stance

        result = system.bayesian_update(
            topic_id="topic1",
            message_stance=message_stance,
            intensity=intensity,
            trust_in_sender=trust,
            current_tick=1,
        )

        # Verify formula
        assert np.isclose(result.delta_stance, expected_delta, rtol=1e-10)
        assert np.isclose(result.new_position, expected_new_position, rtol=1e-10)

    def test_confidence_boost_on_consistency(self):
        """Test confidence increases when message aligns with current stance."""
        system = BayesianBeliefSystem()
        system.initialize_stance("topic1", position=0.5, confidence=0.5)

        old_conf = system.get_confidence("topic1")

        # Message in same direction (positive)
        result = system.bayesian_update(
            topic_id="topic1",
            message_stance=0.8,
            intensity=0.5,
            trust_in_sender=0.7,
            current_tick=1,
        )

        # Confidence should increase
        assert result.new_confidence > old_conf

    def test_confidence_penalty_on_contradiction(self):
        """Test confidence decreases when message contradicts stance."""
        system = BayesianBeliefSystem()
        system.initialize_stance("topic1", position=0.5, confidence=0.5)

        old_conf = system.get_confidence("topic1")

        # Message in opposite direction (negative)
        result = system.bayesian_update(
            topic_id="topic1",
            message_stance=-0.8,
            intensity=0.5,
            trust_in_sender=0.7,
            current_tick=1,
        )

        # Confidence should decrease
        assert result.new_confidence < old_conf

    def test_max_change_clamping(self):
        """Test that stance change per tick is limited."""
        system = BayesianBeliefSystem(max_change_per_tick=0.15)
        system.initialize_stance("topic1", position=0.0, confidence=0.0)

        # Try extreme update
        result = system.bayesian_update(
            topic_id="topic1",
            message_stance=1.0,
            intensity=1.0,
            trust_in_sender=1.0,
            current_tick=1,
        )

        # Change should be clamped to 0.15
        actual_change = abs(result.new_position - result.old_position)
        assert actual_change <= 0.15 + 1e-10

    def test_position_bounds(self):
        """Test stance position stays within [-1, 1]."""
        system = BayesianBeliefSystem()
        system.initialize_stance("topic1", position=0.9, confidence=0.5)

        # Multiple updates pushing toward boundary
        for tick in range(10):
            system.bayesian_update(
                topic_id="topic1",
                message_stance=1.0,
                intensity=1.0,
                trust_in_sender=1.0,
                current_tick=tick,
            )

        final_position = system.get_position("topic1")
        assert -1.0 <= final_position <= 1.0
        # Should be clamped at 1.0
        assert final_position <= 1.0

    def test_high_confidence_resistance(self):
        """Test that high confidence makes agent resistant to change."""
        system = BayesianBeliefSystem()

        # Low confidence agent
        system.initialize_stance("topic1", position=0.0, confidence=0.1)
        result_low = system.bayesian_update(
            topic_id="topic1",
            message_stance=1.0,
            intensity=1.0,
            trust_in_sender=1.0,
            current_tick=1,
        )
        change_low = abs(result_low.new_position - result_low.old_position)

        # High confidence agent
        system2 = BayesianBeliefSystem()
        system2.initialize_stance("topic1", position=0.0, confidence=0.9)
        result_high = system2.bayesian_update(
            topic_id="topic1",
            message_stance=1.0,
            intensity=1.0,
            trust_in_sender=1.0,
            current_tick=1,
        )
        change_high = abs(result_high.new_position - result_high.old_position)

        # Low confidence should change more
        assert change_low > change_high

    def test_openness_moderation(self):
        """Test that openness trait moderates updates."""
        # Low openness agent
        system_low = BayesianBeliefSystem(
            traits=BigFiveTraits(openness=0.2)
        )
        system_low.initialize_stance("topic1", position=0.0, confidence=0.5)
        result_low = system_low.bayesian_update(
            topic_id="topic1",
            message_stance=1.0,
            intensity=1.0,
            trust_in_sender=1.0,
            current_tick=1,
        )
        change_low = abs(result_low.new_position - result_low.old_position)

        # High openness agent
        system_high = BayesianBeliefSystem(
            traits=BigFiveTraits(openness=0.8)
        )
        system_high.initialize_stance("topic1", position=0.0, confidence=0.5)
        result_high = system_high.bayesian_update(
            topic_id="topic1",
            message_stance=1.0,
            intensity=1.0,
            trust_in_sender=1.0,
            current_tick=1,
        )
        change_high = abs(result_high.new_position - result_high.old_position)

        # High openness should change more
        assert change_high > change_low

    def test_serialization(self):
        """Test round-trip serialization."""
        system = BayesianBeliefSystem()
        system.initialize_stance("topic1", position=0.5, confidence=0.6)
        system.bayesian_update(
            topic_id="topic1",
            message_stance=0.8,
            intensity=0.5,
            trust_in_sender=0.7,
            current_tick=1,
        )

        data = system.to_dict()
        restored = BayesianBeliefSystem.from_dict(data)

        assert restored.get_position("topic1") == system.get_position("topic1")
        assert restored.get_confidence("topic1") == system.get_confidence("topic1")


class TestBeliefDistance:
    """Test belief distance computation."""

    def test_identical_stances(self):
        """Distance should be 0 for identical stances."""
        dist = compute_belief_distance(0.5, 0.5, 0.5, 0.5)
        assert dist == 0.0

    def test_opposite_stances(self):
        """Distance should be maximum for opposite stances."""
        dist = compute_belief_distance(-1.0, 1.0, 0.0, 0.0)
        assert dist == 2.0

    def test_confidence_amplification(self):
        """Higher confidence should amplify perceived difference."""
        base_dist = compute_belief_distance(0.0, 0.5, 0.0, 0.0)
        high_conf_dist = compute_belief_distance(0.0, 0.5, 1.0, 1.0)

        # Higher confidence = higher effective distance
        assert high_conf_dist > base_dist


class TestStanceState:
    """Test Pydantic StanceState model."""

    def test_valid_stance(self):
        """Test valid stance creation."""
        stance = StanceState(
            topic_id="test",
            position=0.5,
            confidence=0.8,
        )
        assert stance.position == 0.5
        assert stance.confidence == 0.8

    def test_valid_range_enforcement(self):
        """Test that Pydantic enforces valid ranges at construction."""
        # Values within range work
        stance = StanceState(
            topic_id="test",
            position=0.5,
            confidence=0.8,
        )
        assert stance.position == 0.5
        assert stance.confidence == 0.8

        # Values outside range fail validation at construction
        with pytest.raises(Exception):
            StanceState(
                topic_id="test",
                position=1.5,  # Exceeds maximum
                confidence=0.5,
            )


class TestSanitizeFloat:
    """Test NaN/Inf sanitization helper."""

    def test_normal_value_passes_through(self):
        """Normal values should be unchanged."""
        assert _sanitize_float(0.5) == 0.5
        assert _sanitize_float(-0.5) == -0.5
        assert _sanitize_float(0.0) == 0.0

    def test_nan_replaced_with_default(self):
        """NaN should be replaced with default value."""
        result = _sanitize_float(float('nan'), default=0.5)
        assert not np.isnan(result)
        assert result == 0.5

    def test_inf_replaced_with_default(self):
        """Inf should be replaced with default value."""
        result = _sanitize_float(float('inf'), default=0.5, bounds=(0.0, 1.0))
        assert not np.isinf(result)
        assert result == 0.5

    def test_neg_inf_replaced_with_default(self):
        """-Inf should be replaced with default value."""
        result = _sanitize_float(float('-inf'), default=0.5, bounds=(0.0, 1.0))
        assert not np.isinf(result)
        assert result == 0.5

    def test_bounds_enforcement(self):
        """Values should be clipped to bounds."""
        assert _sanitize_float(1.5, bounds=(-1.0, 1.0)) == 1.0
        assert _sanitize_float(-1.5, bounds=(-1.0, 1.0)) == -1.0

    def test_bayesian_update_with_nan_inputs(self):
        """Bayesian update should handle NaN inputs gracefully."""
        system = BayesianBeliefSystem()
        system.initialize_stance("topic1", position=0.0, confidence=0.5)

        # Update with NaN message stance - should use default (0.0)
        result = system.bayesian_update(
            topic_id="topic1",
            message_stance=float('nan'),
            intensity=0.5,
            trust_in_sender=0.6,
            current_tick=1,
        )

        # Should not crash, position should change minimally
        assert not np.isnan(result.new_position)
        assert -1.0 <= result.new_position <= 1.0

    def test_bayesian_update_with_inf_inputs(self):
        """Bayesian update should handle Inf inputs gracefully."""
        system = BayesianBeliefSystem()
        system.initialize_stance("topic1", position=0.0, confidence=0.5)

        # Update with Inf intensity - should be clamped
        result = system.bayesian_update(
            topic_id="topic1",
            message_stance=0.5,
            intensity=float('inf'),
            trust_in_sender=0.6,
            current_tick=1,
        )

        # Should not crash, intensity should be clamped to [0, 1]
        assert not np.isnan(result.new_position)
        assert not np.isinf(result.new_position)
        assert -1.0 <= result.new_position <= 1.0

    def test_initialize_stance_with_nan(self):
        """Stance initialization should sanitize NaN inputs."""
        system = BayesianBeliefSystem()

        stance = system.initialize_stance(
            "topic1",
            position=float('nan'),
            confidence=float('nan'),
        )

        # Should use defaults
        assert stance.position == 0.0
        assert stance.confidence == 0.5

    def test_initialize_stance_with_inf(self):
        """Stance initialization should sanitize Inf inputs."""
        system = BayesianBeliefSystem()

        stance = system.initialize_stance(
            "topic1",
            position=float('inf'),
            confidence=float('inf'),
        )

        # Should use defaults within bounds
        assert -1.0 <= stance.position <= 1.0
        assert 0.0 <= stance.confidence <= 1.0
