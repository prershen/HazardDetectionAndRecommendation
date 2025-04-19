import os
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from core.config import settings
from pymongo.collection import Collection
from simple_agent.models.simple_agent_models import Message, SimpleAgentState
from datetime import datetime
from bson.objectid import ObjectId

# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)


class SimpleAgentController:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        # Initialize LangGraph agent
        self.graph = self._build_agent_graph()
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph state machine for the agent."""
        # Define the agent state machine
        builder = StateGraph(SimpleAgentState)
        
        # Define the agent nodes
        builder.add_node("retrieve_information", self._retrieve_information)
        builder.add_node("generate_response", self._generate_response)
        
        # Define edges
        builder.add_edge("retrieve_information", "generate_response")
        builder.add_edge("generate_response", END)
        
        # Set entry point
        builder.set_entry_point("retrieve_information")
        
        # Compile the graph
        return builder.compile()
    
    async def _retrieve_information(self, state: SimpleAgentState, db: Collection) -> SimpleAgentState:
        """Retrieve information based on the user's query."""
        # Get the last message from the user
        last_message = state.conversation_history[-1]
        
        # Connect to collections
        spaces_collection = db["spaces"]
        space_logs_collection = db["space_logs"]
        patients_collection = db["patients"]
        
        # Extract user ID
        user_id = state.user_id
        
        # Retrieve all spaces for this user
        spaces = list(spaces_collection.find({"user_id": user_id}))
        
        # Retrieve all patients for this user
        patients = list(patients_collection.find({"user_id": user_id}))
        
        # Create a lookup dictionary for spaces and patients by ID for quicker access
        space_lookup = {str(space["_id"]): space for space in spaces}
        patient_lookup = {str(patient["_id"]): patient for patient in patients}
        
        # Process space logs with proper ID handling
        space_logs = {}
        
        # For each space, get logs
        for space in spaces:
            space_id = space["_id"]
            space_id_str = str(space_id)
            space_name = space.get('space_name', 'Unknown')
            
            # Try both ObjectId and string formats for space_id
            logs = list(space_logs_collection.find({"space_id": space_id}))
            
            # If no logs found with ObjectId, try with string
            if not logs and isinstance(space_id, ObjectId):
                logs = list(space_logs_collection.find({"space_id": space_id_str}))
            
            # If still no logs found and we have a string, try converting to ObjectId
            if not logs and isinstance(space_id_str, str):
                try:
                    space_id_obj = ObjectId(space_id_str)
                    logs = list(space_logs_collection.find({"space_id": space_id_obj}))
                except:
                    # Invalid ObjectId format, continue
                    pass
            
            if logs:
                # Sort logs by creation date (newest first)
                sorted_logs = sorted(logs, key=lambda x: x.get("created_at", datetime.min), reverse=True)
                space_logs[space_id_str] = sorted_logs
                
                # Print debug info for the first log
                first_log = sorted_logs[0]
                print(f"Found log for space '{space_name}': score={first_log.get('score')}")
            else:
                print(f"No logs found for space '{space_name}' with ID {space_id}")
                space_logs[space_id_str] = []
        
        # Build patient-space relationship mapping
        patient_spaces = {}
        for patient in patients:
            patient_id = str(patient["_id"])
            patient_name = patient.get("patient_name", "Unknown")
            patient_relationship = patient.get("patient_relationship", "Unknown").lower()
            
            associated_spaces = []
            
            # Find all spaces with this patient in patient_ids
            for space in spaces:
                space_id = str(space["_id"])
                if "patient_ids" in space and patient_id in space["patient_ids"]:
                    space_info = {
                        "space_id": space_id,
                        "space_name": space.get("space_name", "Unknown"),
                        "description": space.get("description", ""),
                    }
                    
                    # Add safety information from logs if available
                    if space_id in space_logs and space_logs[space_id]:
                        latest_log = space_logs[space_id][0]
                        space_info["safety_score"] = latest_log.get("score")
                        space_info["log_date"] = latest_log.get("created_at")
                        space_info["hazard_list"] = latest_log.get("hazard_list", {})
                        space_info["recommendations"] = latest_log.get("recommendations", [])
                    
                    associated_spaces.append(space_info)
            
            patient_spaces[patient_id] = {
                "patient_name": patient_name,
                "patient_relationship": patient_relationship,
                "spaces": associated_spaces
            }
        
        # Store all information in the state's context
        state.context["spaces"] = spaces
        state.context["patients"] = patients
        state.context["space_logs"] = space_logs
        state.context["patient_spaces"] = patient_spaces
        state.context["space_lookup"] = space_lookup
        state.context["patient_lookup"] = patient_lookup
        
        return state
    
    def _generate_response(self, state: SimpleAgentState) -> SimpleAgentState:
        """Generate a response based on the retrieved information."""
        # Get the last message from the user
        last_message = state.conversation_history[-1]
        query = last_message.content.lower()
        
        # Extract information from context
        spaces = state.context.get("spaces", [])
        patients = state.context.get("patients", [])
        space_logs = state.context.get("space_logs", {})
        patient_spaces = state.context.get("patient_spaces", {})
        
        # Check if the query is about a specific patient by relationship
        is_relationship_query = False
        target_relationship = None
        relationship_terms = {
            "father": ["father", "dad", "daddy", "papa"],
            "mother": ["mother", "mom", "mommy", "mama"],
            "brother": ["brother", "bro"],
            "sister": ["sister", "sis"],
            "grandfather": ["grandfather", "grandpa", "granddad"],
            "grandmother": ["grandmother", "grandma", "granny"],
            "son": ["son"],
            "daughter": ["daughter"],
            "husband": ["husband"],
            "wife": ["wife"]
        }
        
        for relationship, terms in relationship_terms.items():
            if any(term in query for term in terms):
                is_relationship_query = True
                target_relationship = relationship
                break
        
        # Construct context for the model
        context = ""
        
        # If query is about a specific relationship, prioritize that information
        if is_relationship_query and target_relationship:
            context += f"INFORMATION ABOUT YOUR {target_relationship.upper()}:\n"
            found_match = False
            
            for patient_id, patient_data in patient_spaces.items():
                patient_relationship = patient_data["patient_relationship"].lower()
                
                # Check if this patient matches the relationship we're looking for
                relationship_match = False
                
                # Direct match
                if patient_relationship == target_relationship:
                    relationship_match = True
                
                # Check for alternative terms
                for rel, terms in relationship_terms.items():
                    if patient_relationship in terms:
                        if rel == target_relationship:
                            relationship_match = True
                            break
                
                if relationship_match:
                    found_match = True
                    patient_name = patient_data["patient_name"]
                    context += f"Name: {patient_name}\n"
                    context += f"Relationship: {patient_relationship}\n"
                    
                    # Add information about spaces this patient is associated with
                    associated_spaces = patient_data["spaces"]
                    if associated_spaces:
                        context += f"Associated Spaces ({len(associated_spaces)}):\n"
                        for i, space_info in enumerate(associated_spaces, 1):
                            space_name = space_info["space_name"]
                            context += f"  {i}. {space_name}\n"
                            
                            # Add safety score information
                            if "safety_score" in space_info:
                                safety_score = space_info["safety_score"]
                                context += f"     Safety Score: {safety_score}\n"
                                
                                # Add hazard information
                                hazard_list = space_info.get("hazard_list", {})
                                if hazard_list:
                                    high_hazards = hazard_list.get("high_priority", [])
                                    medium_hazards = hazard_list.get("medium_priority", [])
                                    low_hazards = hazard_list.get("low_priority", [])
                                    
                                    # Count total hazards
                                    total_hazards = len(high_hazards) + len(medium_hazards) + len(low_hazards)
                                    context += f"     Total Hazards: {total_hazards}\n"
                                    
                                    # List hazards by priority
                                    if high_hazards:
                                        context += f"     High Priority Hazards: {', '.join(high_hazards)}\n"
                                    if medium_hazards:
                                        context += f"     Medium Priority Hazards: {', '.join(medium_hazards)}\n"
                                    if low_hazards:
                                        context += f"     Low Priority Hazards: {', '.join(low_hazards)}\n"
                                
                                # Add recommendations
                                recommendations = space_info.get("recommendations", [])
                                if recommendations:
                                    context += f"     Recommendations: {', '.join(recommendations)}\n"
                            else:
                                context += f"     No safety information available for this space.\n"
                    else:
                        context += "Not associated with any spaces.\n"
                    
                    context += "\n"
            
            if not found_match:
                context += f"No patient with relationship '{target_relationship}' found in your data.\n\n"
        
        # Check if query is about a specific space/room
        is_space_query = False
        target_space = None
        
        # List of common terms for rooms/spaces
        space_terms = ["room", "space", "area", "bathroom", "bedroom", "kitchen", "living room", "hallway"]
        
        # Check if query mentions specific spaces
        for space in spaces:
            space_name = space.get("space_name", "").lower()
            if space_name in query:
                is_space_query = True
                target_space = space
                break
        
        # Check if query mentions general space terms
        if not is_space_query:
            for term in space_terms:
                if term in query:
                    is_space_query = True
                    break
        
        # If query is specifically about a space, prioritize that information
        if is_space_query and target_space:
            space_id = str(target_space["_id"])
            space_name = target_space.get("space_name", "Unknown")
            
            context += f"INFORMATION ABOUT {space_name.upper()}:\n"
            context += f"Description: {target_space.get('description', 'No description')}\n"
            
            # Find patients associated with this space
            associated_patients = []
            if "patient_ids" in target_space:
                for patient_id in target_space["patient_ids"]:
                    for patient in patients:
                        if str(patient["_id"]) == patient_id:
                            associated_patients.append(patient.get("patient_name", "Unknown"))
            
            if associated_patients:
                context += f"Associated patients: {', '.join(associated_patients)}\n"
            
            # Add safety information
            if space_id in space_logs and space_logs[space_id]:
                latest_log = space_logs[space_id][0]
                safety_score = latest_log.get("score")
                context += f"Latest safety score: {safety_score}\n"
                
                log_date = latest_log.get("created_at")
                if log_date:
                    context += f"Assessment date: {log_date}\n"
                
                # Add hazards
                hazard_list = latest_log.get("hazard_list", {})
                if hazard_list:
                    context += "Hazards:\n"
                    for priority in ["high_priority", "medium_priority", "low_priority"]:
                        hazards = hazard_list.get(priority, [])
                        if hazards:
                            priority_label = priority.replace("_", " ").title()
                            context += f"  {priority_label}: {', '.join(hazards)}\n"
                
                # Add recommendations
                recommendations = latest_log.get("recommendations", [])
                if recommendations:
                    context += f"Recommendations: {', '.join(recommendations)}\n"
            else:
                context += "No safety information available for this space.\n"
            
            context += "\n"
        
        # Add general space information if not already covered
        if not is_space_query or not target_space:
            context += "ALL SPACES INFORMATION:\n"
            for space in spaces:
                space_id = str(space["_id"])
                context += f"Space: {space.get('space_name', 'Unknown')}\n"
                context += f"Description: {space.get('description', 'No description')}\n"
                
                # Add patient associations
                if "patient_ids" in space and space["patient_ids"]:
                    patient_names = []
                    for patient_id in space["patient_ids"]:
                        for patient in patients:
                            if str(patient["_id"]) == patient_id:
                                patient_names.append(patient["patient_name"])
                    if patient_names:
                        context += f"Associated patients: {', '.join(patient_names)}\n"
                
                # Add latest log information
                if space_id in space_logs and space_logs[space_id]:
                    latest_log = space_logs[space_id][0]
                    safety_score = latest_log.get('score')
                    if safety_score is None:
                        safety_score = "No score available"
                    context += f"Latest safety score: {safety_score}\n"
                    
                    # Add hazards
                    hazard_list = latest_log.get("hazard_list", {})
                    if hazard_list:
                        context += "Hazards:\n"
                        for priority in ["high_priority", "medium_priority", "low_priority"]:
                            hazards = hazard_list.get(priority, [])
                            if hazards:
                                priority_label = priority.replace("_", " ").title()
                                context += f"  {priority_label}: {', '.join(hazards)}\n"
                else:
                    context += "No safety logs available for this space.\n"
                
                context += "\n"
        
        # Add general patient information if no specific relationship was queried
        if not is_relationship_query:
            context += "ALL PATIENTS INFORMATION:\n"
            for patient in patients:
                context += f"Patient: {patient.get('patient_name', 'Unknown')}\n"
                context += f"Age: {patient.get('patient_age', 'Unknown')}\n"
                context += f"Condition: {patient.get('patient_condition', 'Unknown')}\n"
                context += f"Relationship: {patient.get('patient_relationship', 'Unknown')}\n"
                
                if "medical_history" in patient and patient["medical_history"]:
                    context += f"Medical History: {patient['medical_history']}\n"
                
                patient_id = str(patient["_id"])
                if patient_id in patient_spaces:
                    spaces_info = patient_spaces[patient_id]["spaces"]
                    if spaces_info:
                        space_names = [space["space_name"] for space in spaces_info]
                        context += f"Associated spaces: {', '.join(space_names)}\n"
                
                context += "\n"
        
        # Generate the response
        prompt = f"""
        You are an AI assistant for a healthcare safety application.
        Your role is to help users understand safety risks in their spaces (rooms) and provide recommendations to improve safety for patients.
        
        USER QUERY: {last_message.content}
        
        AVAILABLE CONTEXT:
        {context}
        
        IMPORTANT RESPONSE GUIDELINES:
        1. Use short sentences and bullet points where appropriate
        2. ONLY provide information that is directly supported by the data provided above and facts
        3. DO NOT hallucinate information that isn't present in the context
        4. Keep your response brief - aim for 5-7 sentences maximum if necessary unless listing specific items
        5. Focus only on answering the user's specific question
        6. When discussing safety scores, always mention the specific value when available
        7. When the user asks about a person by relationship (father, dad, etc.), make sure to address that specific person in your response
        8. If the user asks if someone is "safe", analyze their safety scores and hazards in the spaces they're associated with
        
        Based on the provided context, give a response:
        """
        
        response = self.model.generate_content(prompt)
        
        # Update the conversation history
        state.conversation_history.append(Message(role="assistant", content=response.text))
        
        return state
    
    async def process_conversation(self, user_id: str, query: str, history: Any, db: Collection) -> Tuple[str, Any]:
        """Process a conversation turn and return the agent's response."""
        # Convert history to Message objects if needed
        conversation_history = []
        if history:
            if isinstance(history, list):
                for item in history:
                    if isinstance(item, Message):
                        conversation_history.append(item)
                    elif isinstance(item, dict) and 'role' in item and 'content' in item:
                        conversation_history.append(Message(role=item['role'], content=item['content']))
        
        # Add the new user message
        conversation_history.append(Message(role="user", content=query))
        
        # Initialize state
        state = SimpleAgentState(
            conversation_history=conversation_history,
            user_id=user_id
        )
        
        try:
            # Process information retrieval
            state = await self._retrieve_information(state, db)
            
            # Generate response
            state = self._generate_response(state)
            
            # Return the last message from the assistant
            for message in reversed(state.conversation_history):
                if message.role == "assistant":
                    return message.content, state.conversation_history
            
            # Fallback if no assistant message found
            return "I'm sorry, I couldn't process your request.", state.conversation_history
        
        except Exception as e:
            print(f"Error in agent processing: {str(e)}")
            return f"I encountered an error while processing your request. Please try again.", conversation_history 