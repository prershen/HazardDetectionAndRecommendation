from fastapi import APIRouter, HTTPException, Depends, status
from agent.models.agent_models import ConversationInput, ConversationOutput, Message
from agent.controllers.agent_controller import AgentController
from user.db.mongodb import get_db
from pymongo.collection import Collection
from user.utils.auth import get_current_user
from typing import List

router = APIRouter()
agent_controller = AgentController()


@router.post("/conversation", response_model=ConversationOutput)
async def have_conversation(
    conversation: ConversationInput,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    """
    Endpoint for conversational interaction with the AI agent.
    The agent will analyze the query, retrieve relevant data, and respond appropriately.
    """
    try:
        # For security, ensure the user_id matches the authenticated user
        if conversation.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User ID mismatch with authenticated user"
            )
        print(conversation)
        # Process the conversation
        response_text, updated_history = await agent_controller.process_conversation(
            user_id=current_user_id,
            query=conversation.query,
            history=conversation.history,
            db=db
        )
        
        # Return the response
        return ConversationOutput(
            response=response_text,
            history=updated_history
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversation processing failed: {str(e)}"
        ) 