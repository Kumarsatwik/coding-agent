from __future__ import annotations
import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema

class ToolKind(str,Enum):
    """Enumeration of tool types."""
    READ='read'
    WRITE='write'
    SHELL='shell'
    NETWORK='network'
    MEMORY='memory'
    MCP='mcp'

@dataclass
class ToolInvocation:
    """Tool invocation with parameters and working directory."""
    params:dict[str,Any]
    cwd:Path

@dataclass
class ToolResult:
    """Outcome of a tool execution."""
    success:bool
    output:str
    error:str | None = None
    metadata:dict[str,Any] = field(default_factory=dict)



@dataclass
class ToolConfirmation(ToolInvocation):
    """Request for user confirmation of a mutating tool."""
    tool_name:str
    params:dict[str,Any]
    description:str



@dataclass
class ToolResult(BaseModel):
    """Result of a tool execution (BaseModel version)."""
    success:bool
    content:str | None = None


class Tool(abc.ABC):
    """Base class for all tools."""
    name:str = "base_tool"
    description:str = "Base Tool"
    kind:ToolKind = ToolKind.READ
   
    def __init__(self)->None:
        pass
    
    @property
    def schema(self) -> dict[str, Any] | type['BaseModel']:
        """Parameter schema for the tool."""
        raise NotImplementedError("Tool must define schema property or class attribute")

    @abc.abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """Execute the tool with given parameters."""
        pass 

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters against schema."""
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                BaseModel(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation Error")
                    errors.append(f"Parameter '{field}':{msg}")
                
                return errors
            except Exception as e:
                return [str(e)]
        return []
    
    def is_mutating(self, params: dict[str, Any]) -> bool:
        """Check if tool performs a mutating operation."""
        return self.kind in (ToolKind.WRITE, ToolKind.SHELL, ToolKind.NETWORK, ToolKind.MEMORY)
        
    async def get_confirmation(self, invocation: ToolInvocation) -> ToolInvocation | None:
        """Get confirmation request if tool is mutating."""
        if not self.is_mutating(invocation.params):
            return None
        
        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}",
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert tool schema to OpenAI format."""
        schema = self.schema
        
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = model_json_schema(schema, mode='serialization')

            return {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': json_schema.get('properties', {}),
                    'required': json_schema.get('required', []),
                },
            }
        
        if isinstance(schema, dict):
            result = {'name': self.name, 'description': self.description}

            if 'parameters' in schema:
                result['parameters'] = schema['parameters']
            else:
                result['parameters'] = schema

            return result
            
        raise ValueError(f"Unsupported schema type {self.name}")
