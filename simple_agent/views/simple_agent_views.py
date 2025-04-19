from fastapi import APIRouter, HTTPException, Depends, status
from simple_agent.models.simple_agent_models import ConversationInput, ConversationOutput
from simple_agent.controllers.simple_agent_controller import SimpleAgentController
from user.db.mongodb import get_db
from pymongo.collection import Collection
from user.utils.auth import get_current_user
from typing import List, Any

router = APIRouter()
simple_agent_controller = SimpleAgentController()


@router.post("/simple_conversation", response_model=ConversationOutput)
async def have_simple_conversation(
    conversation: ConversationInput,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    """
    Endpoint for simple conversational interaction with the AI agent.
    The agent will retrieve relevant data about spaces, patients, and safety information
    and respond to user queries in a conversational manner.
    """
    try:
        print("conversation: ", conversation)
        # For security, ensure the user_id matches the authenticated user
        if conversation.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User ID mismatch with authenticated user"
            )
        
        # Process the conversation
        response_text, updated_history = await simple_agent_controller.process_conversation(
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
        print(f"Error in simple conversation endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversation processing failed: {str(e)}"
        ) 