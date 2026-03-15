from __future__ import annotations

from agent_search.nodes.base import BaseNodesMixin
from agent_search.nodes.initial_answer import InitialAnswerMixin
from agent_search.nodes.logging import LoggingMixin
from agent_search.nodes.research_agent import ResearchAgentMixin
from agent_search.nodes.refinement import RefinementMixin
from agent_search.nodes.routing import RoutingMixin
from agent_search.nodes.synthesis import SynthesisMixin
from agent_search.nodes.validation import ValidationMixin


class AgentSearchNodes(
    RoutingMixin,
    InitialAnswerMixin,
    ResearchAgentMixin,
    RefinementMixin,
    SynthesisMixin,
    ValidationMixin,
    LoggingMixin,
    BaseNodesMixin,
):
    pass
