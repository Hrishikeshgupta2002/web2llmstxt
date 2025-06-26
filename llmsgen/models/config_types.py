"""
Dataclasses for AI model configurations and related types.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """Configuration for AI models with enhanced metadata"""
    provider: str
    model_id: str
    display_name: str
    description: str
    max_tokens: int = 8192
    temperature: float = 0.7
    supports_vision: bool = False
    cost_per_1k_tokens: float = 0.0
    status: str = "unknown"  # online, offline, unknown, available
    tags: List[str] = field(default_factory=list)
    pulls: str = ""
    size_info: str = ""
    is_available_remote: bool = False
    estimated_ram_gb: float = 0.0  # Estimated RAM requirement
