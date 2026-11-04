# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All rights reserved.
# MIT license: see NEXENT-LICENSE.txt. Exact four class excerpts from Nexent
# d289f67ef11558bbdb5d9a750dbb8fefcbce8cfc. Imports and None-only helper added
# locally for isolated schema checking, NOT the Nexent importer/runtime.
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

def _validated_context_policy(value):
    if value is not None:
        raise ValueError("This offline check only supports the default context policy")
    return None

class ToolConfig(BaseModel):
    class_name: str = Field(description="Tool class name")
    name: Optional[str] = Field(description="Tool name")
    description: Optional[str] = Field(description="Tool description", default=None)
    description_zh: Optional[str] = Field(description="Chinese tool description", default=None)
    inputs: Optional[str] = Field(description="Tool inputs", default=None)
    output_type: Optional[str] = Field(description="Tool output type", default=None)
    params: Dict[str, Any] = Field(description="Initialization parameters", default=None)
    source: str = Field(description="Tool source, can be local or mcp", default="local")
    usage: Optional[str] = Field(description="MCP server name", default=None)
    metadata: Optional[Dict[str, Any]] = Field(description="Metadata", default=None)
    labels: Optional[List[str]] = Field(description="Tool labels for filtering", default=None)


class ExportAndImportAgentInfo(BaseModel):
    agent_id: int
    tenant_id: Optional[str] = None
    name: str
    display_name: Optional[str] = None
    description: str
    author: Optional[str] = None
    max_steps: int
    requested_output_tokens: Optional[int] = Field(default=None, gt=0)
    is_main_agent: bool = True
    provide_run_summary: bool
    allow_chat_metadata: bool = False
    enable_protocol_repair_retry: bool = False
    verification_config: Optional[Dict[str, Any]] = None
    context_policy: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    enabled: bool
    tools: List[ToolConfig]
    managed_agents: List[int]
    model_ids: Optional[List[int]] = None
    model_names: Optional[List[str]] = None
    business_logic_model_id: Optional[int] = None
    business_logic_model_name: Optional[str] = None
    skill_names: Optional[List[str]] = None
    # v2.6.0 per-agent model inference param overrides (mirrors AgentInfo column).
    model_params_override: Optional[Dict[str, Any]] = None
    greeting_message: Optional[str] = None
    example_questions: Optional[List[str]] = None

    @field_validator("context_policy")
    @classmethod
    def validate_context_policy(cls, value):
        return _validated_context_policy(value)

    class Config:
        arbitrary_types_allowed = True


class MCPInfo(BaseModel):
    mcp_server_name: str
    mcp_url: str


class ExportAndImportDataFormat(BaseModel):
    agent_id: int
    agent_info: Dict[str, ExportAndImportAgentInfo]
    mcp_info: List[MCPInfo]
