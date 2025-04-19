from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A chat message."""
    role: str
    content: str


class ConversationInput(BaseModel):
    """Input for the conversation endpoint."""
    query: str
    user_id: str
    history: Any = Field(default_factory=list)


class ConversationOutput(BaseModel):
    """Output from the conversation endpoint."""
    response: str
    history: Any = Field(default_factory=list)


class SimpleAgentState(BaseModel):
    """Internal state of the simple agent."""
    conversation_history: List[Message] = Field(default_factory=list)
    user_id: str
    context: Dict[str, Any] = Field(default_factory=dict) 