"""Pydantic schemas for OpenAI-compatible request and response objects."""

import time
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None
    strict: Optional[bool] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


ToolChoice = Union[Literal["auto", "none", "required"], ToolChoiceObject]


class ResponseFormatText(BaseModel):
    type: Literal["text"] = "text"


class ResponseFormatJSON(BaseModel):
    type: Literal["json_object"] = "json_object"


class JSONSchemaDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    schema_: Optional[dict] = Field(None, alias="schema")
    strict: Optional[bool] = None

    model_config = {"populate_by_name": True}


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchemaDefinition


ResponseFormat = Union[ResponseFormatText, ResponseFormatJSON, ResponseFormatJSONSchema]


# ---------------------------------------------------------------------------
# Chat message types
# ---------------------------------------------------------------------------

class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Optional[Union[str, list]] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool result messages
    name: Optional[str] = None


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    response_format: Optional[ResponseFormat] = None
    stop: Optional[Union[str, list[str]]] = None
    n: int = 1

    model_config = {"extra": "ignore"}  # Accept and ignore unknown fields


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Optional[Usage] = None


class ChatChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "installm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str]]
    dimensions: Optional[int] = None
    encoding_format: Literal["float", "base64"] = "float"

    model_config = {"extra": "ignore"}


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------

class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, list]
    instructions: Optional[str] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    stream: bool = False
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    model_config = {"extra": "ignore"}
