"""L4: Social Interaction Layer.

This layer handles agent relationships and social dynamics:
- Relationship network
- Information propagation with cognitive distortion
- Group dynamics detection (polarization, opinion leaders)
"""

from utopia.layer4_social.relationships import RelationshipMap, RelationshipEdge, RelationshipDelta
from utopia.layer4_social.propagator import InformationPropagator, Message
from utopia.layer4_social.dynamics import GroupDynamicsDetector, PolarizationReport

__all__ = [
    "RelationshipMap",
    "RelationshipEdge",
    "RelationshipDelta",
    "InformationPropagator",
    "Message",
    "GroupDynamicsDetector",
    "PolarizationReport",
]
