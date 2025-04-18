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
    history: Optional[List[Message]] = Field(default_factory=list)


class ConversationOutput(BaseModel):
    """Output from the conversation endpoint."""
    response: str
    history: List[Message]


class AgentState(BaseModel):
    """Internal state of the agent."""
    conversation_history: List[Message] = Field(default_factory=list)
    user_id: str
    current_context: Dict[str, Any] = Field(default_factory=dict)
    patient_context: Optional[Dict[str, Any]] = None
    space_context: Optional[Dict[str, Any]] = None 