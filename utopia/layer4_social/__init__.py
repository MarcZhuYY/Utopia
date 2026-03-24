"""L4: Social Interaction Layer.

This layer handles agent relationships and social dynamics:
- Relationship network with homophily
- Information propagation with cognitive distortion
- Group dynamics detection (polarization, opinion leaders)
- Vectorized propagation for performance
"""

from utopia.layer4_social.relationships import RelationshipMap, RelationshipEdge, RelationshipDelta
from utopia.layer4_social.homophily import (
    HomophilyEngine,
    HomophilyConfig,
    EchoChamberAnalyzer,
    compute_affinity_matrix,
    compute_trust_delta_matrix,
)
from utopia.layer4_social.dynamics import GroupDynamicsDetector, PolarizationReport
from utopia.layer4_social.social_network_tensor import SocialTensorGraph

__all__ = [
    "RelationshipMap",
    "RelationshipEdge",
    "RelationshipDelta",
    "HomophilyEngine",
    "HomophilyConfig",
    "EchoChamberAnalyzer",
    "compute_affinity_matrix",
    "compute_trust_delta_matrix",
    "GroupDynamicsDetector",
    "PolarizationReport",
    "SocialTensorGraph",
]
