import os
from typing import List, Dict, Any, Optional, Tuple, Annotated, TypedDict
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from core.config import settings
from pymongo.collection import Collection
from agent.models.agent_models import Message, AgentState
from datetime import datetime, timedelta
import re
from bson.objectid import ObjectId

# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)


class AgentController:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
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
        builder = StateGraph(AgentState)
        
        # Define the agent nodes
        builder.add_node("understand_query", self._understand_query)
        builder.add_node("retrieve_patient_info", self._retrieve_patient_info)
        builder.add_node("retrieve_space_info", self._retrieve_space_info)
        builder.add_node("generate_response", self._generate_response)
        
        # Define edges
        builder.add_edge("understand_query", "retrieve_patient_info")
        builder.add_edge("retrieve_patient_info", "retrieve_space_info")
        builder.add_edge("retrieve_space_info", "generate_response")
        builder.add_edge("generate_response", END)
        
        # Set entry point
        builder.set_entry_point("understand_query")
        
        # Compile the graph
        return builder.compile()
    
    def _understand_query(self, state: AgentState) -> AgentState:
        """Understand the user query and determine what information to retrieve."""
        # Get the last message from the user
        last_message = state.conversation_history[-1]
        
        # Use Gemini to understand the query
        prompt = f"""
        You are an AI assistant analyzing a user query to determine what information to retrieve.
        The query is: {last_message.content}
        
        Analyze whether this query is about:
        1. Patient information (medical history, condition, age)
        2. Space information (safety scores, hazards, recommendations)
        3. Space history or logs over time
        4. Space comparison between multiple spaces
        5. Date-specific space information (for a particular date)
        6. Temporal comparison of a single space (e.g., comparing today vs yesterday, or before vs after)
        7. Finding optimal space configuration (best safety score or improvement)
        8. General conversation not requiring specific data retrieval
        
        Return your analysis in JSON format with these fields:
        - needs_patient_info: boolean
        - needs_space_info: boolean
        - needs_space_history: boolean
        - needs_space_comparison: boolean
        - needs_date_specific: boolean
        - needs_temporal_comparison: boolean
        - needs_optimal_config: boolean
        - date_mentioned: string (YYYY-MM-DD format) or null
        - reference_date: string (YYYY-MM-DD format) or null (for temporal comparison)
        - space_names: array of string (if specific space names are mentioned)
        - is_improvement_query: boolean (asking how things improved)
        """
        
        response = self.model.generate_content(prompt)
        analysis = response.text
        
        # Update the state with context
        state.current_context["query_analysis"] = analysis
        return state
    
    async def _retrieve_patient_info(self, state: AgentState, db: Collection) -> AgentState:
        """Retrieve patient information from the database if relevant to the query."""
        # Check if we need patient information
        analysis = state.current_context.get("query_analysis", "")
        if "needs_patient_info" in analysis.lower() and "true" in analysis.lower():
            # Get patient data for this user
            patients_collection = db["patients"]
            patients = list(patients_collection.find({"user_id": state.user_id}))
            
            # Store the patient data in the state
            state.patient_context = {"patients": patients}
        
        return state
    
    async def _retrieve_space_info(self, state: AgentState, db: Collection) -> AgentState:
        """Retrieve space information from the database based on the query analysis."""
        # Extract information from analysis
        analysis = state.current_context.get("query_analysis", "")
        needs_space_info = "needs_space_info" in analysis.lower() and "true" in analysis.lower()
        needs_space_history = "needs_space_history" in analysis.lower() and "true" in analysis.lower()
        needs_space_comparison = "needs_space_comparison" in analysis.lower() and "true" in analysis.lower()
        needs_date_specific = "needs_date_specific" in analysis.lower() and "true" in analysis.lower()
        needs_temporal_comparison = "needs_temporal_comparison" in analysis.lower() and "true" in analysis.lower()
        needs_optimal_config = "needs_optimal_config" in analysis.lower() and "true" in analysis.lower()
        is_improvement_query = "is_improvement_query" in analysis.lower() and "true" in analysis.lower()
        
        # Extract date if mentioned
        date_match = re.search(r'"date_mentioned":\s*"([0-9]{4}-[0-9]{2}-[0-9]{2})"', analysis)
        specific_date = date_match.group(1) if date_match else None
        
        # Extract reference date if mentioned (for temporal comparison)
        ref_date_match = re.search(r'"reference_date":\s*"([0-9]{4}-[0-9]{2}-[0-9]{2})"', analysis)
        reference_date = ref_date_match.group(1) if ref_date_match else None
        
        # Extract space names if mentioned
        space_names = []
        space_names_match = re.search(r'"space_names":\s*\[(.*?)\]', analysis)
        if space_names_match:
            space_names_str = space_names_match.group(1)
            space_names = [name.strip(' "\'') for name in space_names_str.split(',') if name.strip()]
        
        # If none of the space-related flags are true, return early
        if not any([needs_space_info, needs_space_history, needs_space_comparison, 
                    needs_date_specific, needs_temporal_comparison, needs_optimal_config]):
            return state
        
        # Get space data for this user
        spaces_collection = db["spaces"]
        spaces = list(spaces_collection.find({"user_id": state.user_id}))
        
        # Filter spaces by name if specified
        if space_names:
            spaces = [space for space in spaces if space.get("space_name", "").lower() in 
                      [name.lower() for name in space_names]]
        
        # Prepare space context
        space_context = {"spaces": spaces}
        
        # If we have spaces, get space logs
        if spaces:
            space_logs_collection = db["space_logs"]
            space_ids = [space["_id"] for space in spaces]
            
            # Different log retrieval strategies based on query type
            if needs_date_specific and specific_date:
                # Parse the specific date
                try:
                    target_date = datetime.strptime(specific_date, "%Y-%m-%d")
                    next_date = target_date + timedelta(days=1)
                    
                    # Get logs from the specific date
                    space_logs = list(space_logs_collection.find({
                        "space_id": {"$in": space_ids},
                        "created_at": {
                            "$gte": target_date,
                            "$lt": next_date
                        }
                    }))
                    space_context["date_specific"] = True
                    space_context["specific_date"] = specific_date
                    
                except (ValueError, TypeError):
                    # If date parsing fails, get all logs
                    space_logs = list(space_logs_collection.find({"space_id": {"$in": space_ids}}))
            else:
                # Get all logs for these spaces
                space_logs = list(space_logs_collection.find({"space_id": {"$in": space_ids}}))
            
            # If we need history, organize logs by space and date
            if needs_space_history:
                # Group logs by space
                space_logs_by_space = {}
                for log in space_logs:
                    space_id_str = str(log["space_id"])
                    if space_id_str not in space_logs_by_space:
                        space_logs_by_space[space_id_str] = []
                    space_logs_by_space[space_id_str].append(log)
                
                # Sort logs by date for each space
                for space_id, logs in space_logs_by_space.items():
                    space_logs_by_space[space_id] = sorted(
                        logs, 
                        key=lambda x: x.get("created_at", datetime.min), 
                        reverse=True
                    )
                
                space_context["space_logs_by_space"] = space_logs_by_space
                space_context["history_requested"] = True
            
            # If we need comparison, ensure we have all relevant logs
            if needs_space_comparison:
                space_context["comparison_requested"] = True
            
            # If we need temporal comparison, organize data accordingly
            if needs_temporal_comparison:
                space_context["temporal_comparison"] = True
                space_context["is_improvement_query"] = is_improvement_query
                
                # Set up comparison dates
                if specific_date:
                    target_date = datetime.strptime(specific_date, "%Y-%m-%d")
                else:
                    # Default to today if no date specified
                    target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                if reference_date:
                    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
                else:
                    # Default to yesterday if no reference date
                    ref_date = target_date - timedelta(days=1)
                
                space_context["current_date"] = target_date.strftime("%Y-%m-%d")
                space_context["previous_date"] = ref_date.strftime("%Y-%m-%d")
                
                # Group logs by space and organize by the two dates
                temporal_comparison = {}
                for space in spaces:
                    space_id_str = str(space["_id"])
                    space_name = space.get("space_name", "Unknown")
                    
                    # Get logs for this space
                    space_logs_for_comparison = [log for log in space_logs if str(log["space_id"]) == space_id_str]
                    
                    # Get current/target date logs
                    target_logs = [log for log in space_logs_for_comparison if 
                                 isinstance(log.get("created_at"), datetime) and
                                 log["created_at"].date() == target_date.date()]
                    
                    # Get reference/previous date logs
                    reference_logs = [log for log in space_logs_for_comparison if 
                                    isinstance(log.get("created_at"), datetime) and
                                    log["created_at"].date() == ref_date.date()]
                    
                    # Store the most recent log for each date if available
                    comparison_data = {
                        "space_name": space_name,
                        "space_id": space_id_str,
                        "current_date": target_date.strftime("%Y-%m-%d"),
                        "previous_date": ref_date.strftime("%Y-%m-%d"),
                        "current_log": sorted(target_logs, key=lambda x: x["created_at"], reverse=True)[0] if target_logs else None,
                        "previous_log": sorted(reference_logs, key=lambda x: x["created_at"], reverse=True)[0] if reference_logs else None
                    }
                    
                    temporal_comparison[space_id_str] = comparison_data
                
                space_context["temporal_comparison_data"] = temporal_comparison
            
            # If we need to find optimal configuration
            if needs_optimal_config:
                space_context["optimal_config_requested"] = True
                
                # Group logs by space to find best score
                optimal_configs = {}
                
                # First pass - find best score per space
                for space in spaces:
                    space_id_str = str(space["_id"])
                    space_name = space.get("space_name", "Unknown")
                    
                    # Get logs for this space
                    space_logs_for_comparison = [log for log in space_logs if str(log["space_id"]) == space_id_str]
                    
                    # Sort by score (descending)
                    valid_logs = [log for log in space_logs_for_comparison if log.get("score") is not None]
                    
                    if valid_logs:
                        # Sort by score (highest first)
                        sorted_logs = sorted(valid_logs, key=lambda x: x.get("score", 0), reverse=True)
                        
                        optimal_configs[space_id_str] = {
                            "space_name": space_name,
                            "space_id": space_id_str,
                            "best_log": sorted_logs[0],
                            "best_score": sorted_logs[0].get("score", 0),
                            "all_logs": sorted_logs
                        }
                
                # Second pass - find global best
                best_space_id = None
                best_score = -1
                
                for space_id, config in optimal_configs.items():
                    if config["best_score"] > best_score:
                        best_score = config["best_score"]
                        best_space_id = space_id
                
                if best_space_id:
                    space_context["global_best_space"] = best_space_id
                
                space_context["optimal_configs"] = optimal_configs
            
            # Store all logs for reference
            space_context["space_logs"] = space_logs
        
        # Store the space data in the state
        state.space_context = space_context
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate a response based on the retrieved information."""
        # Get the last message from the user
        last_message = state.conversation_history[-1]
        
        # Construct context for the model
        context = ""
        
        # Add patient context if available
        if state.patient_context and "patients" in state.patient_context:
            patients = state.patient_context["patients"]
            context += "PATIENT INFORMATION:\n"
            for i, patient in enumerate(patients):
                context += f"Patient {i+1}:\n"
                context += f"  Name: {patient['patient_name']}\n"
                context += f"  Age: {patient['patient_age']}\n"
                context += f"  Condition: {patient['patient_condition']}\n"
                if "medical_history" in patient and patient["medical_history"]:
                    context += f"  Medical History: {patient['medical_history']}\n"
                context += "\n"
        
        # Add space context if available
        if state.space_context:
            spaces = state.space_context.get("spaces", [])
            space_logs = state.space_context.get("space_logs", [])
            
            # Handle regular space information
            if spaces:
                context += "SPACE INFORMATION:\n"
                for i, space in enumerate(spaces):
                    context += f"Space {i+1}:\n"
                    context += f"  Name: {space['space_name']}\n"
                    context += f"  ID: {space['_id']}\n"
                    if "description" in space and space["description"]:
                        context += f"  Description: {space['description']}\n"
                    context += f"  Created At: {space.get('created_at', 'Unknown')}\n"
                    
                    # Find logs for this space
                    related_logs = [log for log in space_logs if str(log["space_id"]) == str(space["_id"])]
                    if related_logs:
                        # Just use the most recent log by default
                        latest_log = sorted(related_logs, key=lambda x: x.get("created_at", ""), reverse=True)[0]
                        context += f"  Latest Safety Score: {latest_log.get('score', 'Unknown')}\n"
                        context += f"  Latest Log Date: {latest_log.get('created_at', 'Unknown')}\n"
                        
                        # Add hazards
                        hazard_list = latest_log.get("hazard_list", {})
                        if hazard_list:
                            context += "  Hazards:\n"
                            for priority in ["high_priority", "medium_priority", "low_priority"]:
                                hazards = hazard_list.get(priority, [])
                                if hazards:
                                    priority_label = priority.replace("_", " ").title()
                                    context += f"    {priority_label}:\n"
                                    for hazard in hazards:
                                        context += f"      - {hazard}\n"
                        
                        # Add recommendations
                        recommendations = latest_log.get("recommendations", [])
                        if recommendations:
                            context += "  Recommendations:\n"
                            for rec in recommendations:
                                context += f"    - {rec}\n"
                    
                    context += "\n"
            
            # Add space history if requested
            if state.space_context.get("history_requested", False):
                space_logs_by_space = state.space_context.get("space_logs_by_space", {})
                
                if space_logs_by_space:
                    context += "SPACE HISTORY:\n"
                    
                    for space in spaces:
                        space_id_str = str(space["_id"])
                        if space_id_str in space_logs_by_space:
                            context += f"History for {space['space_name']}:\n"
                            logs = space_logs_by_space[space_id_str]
                            
                            for i, log in enumerate(logs):
                                context += f"  Log {i+1} (Date: {log.get('created_at', 'Unknown')}):\n"
                                context += f"    Safety Score: {log.get('score', 'Unknown')}\n"
                                
                                # Add hazards
                                hazard_list = log.get("hazard_list", {})
                                if hazard_list:
                                    context += "    Hazards:\n"
                                    for priority in ["high_priority", "medium_priority", "low_priority"]:
                                        hazards = hazard_list.get(priority, [])
                                        if hazards:
                                            priority_label = priority.replace("_", " ").title()
                                            context += f"      {priority_label}: {', '.join(hazards)}\n"
                                
                                # Add recommendations
                                recommendations = log.get("recommendations", [])
                                if recommendations:
                                    context += f"    Recommendations: {', '.join(recommendations)}\n"
                            
                            context += "\n"
            
            # Add date-specific information if requested
            if state.space_context.get("date_specific", False):
                specific_date = state.space_context.get("specific_date", "")
                context += f"SPACE INFORMATION FOR DATE {specific_date}:\n"
                
                for space in spaces:
                    space_id_str = str(space["_id"])
                    context += f"Space: {space['space_name']}\n"
                    
                    # Find logs for this space on the specific date
                    date_logs = [log for log in space_logs 
                                if str(log["space_id"]) == space_id_str and 
                                (isinstance(log.get("created_at"), datetime) and 
                                 log.get("created_at").strftime("%Y-%m-%d") == specific_date)]
                    
                    if date_logs:
                        for log in date_logs:
                            context += f"  Log Time: {log.get('created_at', 'Unknown')}\n"
                            context += f"  Safety Score: {log.get('score', 'Unknown')}\n"
                            
                            # Add hazards
                            hazard_list = log.get("hazard_list", {})
                            if hazard_list:
                                context += "  Hazards:\n"
                                for priority in ["high_priority", "medium_priority", "low_priority"]:
                                    hazards = hazard_list.get(priority, [])
                                    if hazards:
                                        priority_label = priority.replace("_", " ").title()
                                        context += f"    {priority_label}: {', '.join(hazards)}\n"
                            
                            # Add recommendations
                            recommendations = log.get("recommendations", [])
                            if recommendations:
                                context += f"  Recommendations: {', '.join(recommendations)}\n"
                    else:
                        context += f"  No logs found for this date.\n"
                    
                    context += "\n"
            
            # Add space comparison if requested
            if state.space_context.get("comparison_requested", False) and len(spaces) > 1:
                context += "SPACE COMPARISON:\n"
                
                # Get latest logs for each space
                space_latest_logs = {}
                for space in spaces:
                    space_id_str = str(space["_id"])
                    related_logs = [log for log in space_logs if str(log["space_id"]) == space_id_str]
                    if related_logs:
                        latest_log = sorted(related_logs, key=lambda x: x.get("created_at", ""), reverse=True)[0]
                        space_latest_logs[space_id_str] = latest_log
                
                # Compare safety scores
                context += "Safety Score Comparison:\n"
                for space in spaces:
                    space_id_str = str(space["_id"])
                    if space_id_str in space_latest_logs:
                        log = space_latest_logs[space_id_str]
                        context += f"  {space['space_name']}: {log.get('score', 'Unknown')}\n"
                
                # Compare hazard counts
                context += "Hazard Count Comparison:\n"
                for space in spaces:
                    space_id_str = str(space["_id"])
                    if space_id_str in space_latest_logs:
                        log = space_latest_logs[space_id_str]
                        hazard_list = log.get("hazard_list", {})
                        
                        high_count = len(hazard_list.get("high_priority", []))
                        medium_count = len(hazard_list.get("medium_priority", []))
                        low_count = len(hazard_list.get("low_priority", []))
                        total_count = high_count + medium_count + low_count
                        
                        context += f"  {space['space_name']}: {total_count} total hazards "
                        context += f"({high_count} high, {medium_count} medium, {low_count} low)\n"
                
                context += "\n"
            
            # Add temporal comparison if requested
            if state.space_context.get("temporal_comparison", False):
                comparison_data = state.space_context.get("temporal_comparison_data", {})
                current_date = state.space_context.get("current_date", "today")
                previous_date = state.space_context.get("previous_date", "previous day")
                is_improvement_query = state.space_context.get("is_improvement_query", False)
                
                context += f"TEMPORAL COMPARISON (Between {previous_date} and {current_date}):\n"
                
                for space_id, data in comparison_data.items():
                    space_name = data.get("space_name", "Unknown Space")
                    context += f"Comparison for {space_name}:\n"
                    
                    current_log = data.get("current_log")
                    previous_log = data.get("previous_log")
                    
                    if current_log and previous_log:
                        # Compare safety scores
                        current_score = current_log.get("score", 0)
                        previous_score = previous_log.get("score", 0)
                        score_difference = current_score - previous_score
                        
                        context += f"  Safety Score:\n"
                        context += f"    {current_date}: {current_score}\n"
                        context += f"    {previous_date}: {previous_score}\n"
                        context += f"    Change: {score_difference:+.2f}\n"
                        
                        # Compare hazards
                        current_hazards = current_log.get("hazard_list", {})
                        previous_hazards = previous_log.get("hazard_list", {})
                        
                        # Count hazards by priority
                        current_high = len(current_hazards.get("high_priority", []))
                        current_medium = len(current_hazards.get("medium_priority", []))
                        current_low = len(current_hazards.get("low_priority", []))
                        current_total = current_high + current_medium + current_low
                        
                        previous_high = len(previous_hazards.get("high_priority", []))
                        previous_medium = len(previous_hazards.get("medium_priority", []))
                        previous_low = len(previous_hazards.get("low_priority", []))
                        previous_total = previous_high + previous_medium + previous_low
                        
                        context += f"  Hazard Count:\n"
                        context += f"    {current_date}: {current_total} total ({current_high} high, {current_medium} medium, {current_low} low)\n"
                        context += f"    {previous_date}: {previous_total} total ({previous_high} high, {previous_medium} medium, {previous_low} low)\n"
                        
                        # Identify fixed and new hazards
                        fixed_hazards = []
                        new_hazards = []
                        
                        # Check for fixed high priority hazards
                        for hazard in previous_hazards.get("high_priority", []):
                            if hazard not in current_hazards.get("high_priority", []):
                                fixed_hazards.append(f"High: {hazard}")
                        
                        # Check for fixed medium priority hazards
                        for hazard in previous_hazards.get("medium_priority", []):
                            if hazard not in current_hazards.get("medium_priority", []):
                                fixed_hazards.append(f"Medium: {hazard}")
                        
                        # Check for fixed low priority hazards
                        for hazard in previous_hazards.get("low_priority", []):
                            if hazard not in current_hazards.get("low_priority", []):
                                fixed_hazards.append(f"Low: {hazard}")
                        
                        # Check for new high priority hazards
                        for hazard in current_hazards.get("high_priority", []):
                            if hazard not in previous_hazards.get("high_priority", []):
                                new_hazards.append(f"High: {hazard}")
                        
                        # Check for new medium priority hazards
                        for hazard in current_hazards.get("medium_priority", []):
                            if hazard not in previous_hazards.get("medium_priority", []):
                                new_hazards.append(f"Medium: {hazard}")
                                
                        # Check for new low priority hazards
                        for hazard in current_hazards.get("low_priority", []):
                            if hazard not in previous_hazards.get("low_priority", []):
                                new_hazards.append(f"Low: {hazard}")
                        
                        if fixed_hazards:
                            context += f"  Resolved Hazards:\n"
                            for hazard in fixed_hazards:
                                context += f"    - {hazard}\n"
                        
                        if new_hazards:
                            context += f"  New Hazards:\n"
                            for hazard in new_hazards:
                                context += f"    - {hazard}\n"
                    
                    elif current_log:
                        context += f"  No data available for {previous_date} to compare with.\n"
                        context += f"  Current Safety Score ({current_date}): {current_log.get('score', 'Unknown')}\n"
                    
                    elif previous_log:
                        context += f"  No data available for {current_date} to compare with.\n"
                        context += f"  Previous Safety Score ({previous_date}): {previous_log.get('score', 'Unknown')}\n"
                    
                    else:
                        context += f"  No data available for comparison.\n"
                    
                    context += "\n"
            
            # Add optimal configuration information if requested
            if state.space_context.get("optimal_config_requested", False):
                optimal_configs = state.space_context.get("optimal_configs", {})
                global_best_space_id = state.space_context.get("global_best_space")
                
                context += "OPTIMAL SPACE CONFIGURATIONS:\n"
                
                if global_best_space_id and global_best_space_id in optimal_configs:
                    best_config = optimal_configs[global_best_space_id]
                    best_log = best_config.get("best_log", {})
                    
                    context += f"Overall Best Space Configuration:\n"
                    context += f"  Space: {best_config['space_name']}\n"
                    context += f"  Best Safety Score: {best_log.get('score', 'Unknown')}\n"
                    context += f"  Date: {best_log.get('created_at', 'Unknown')}\n"
                    
                    # Add hazards
                    hazard_list = best_log.get("hazard_list", {})
                    if hazard_list:
                        high_hazards = hazard_list.get("high_priority", [])
                        medium_hazards = hazard_list.get("medium_priority", [])
                        low_hazards = hazard_list.get("low_priority", [])
                        
                        total_hazards = len(high_hazards) + len(medium_hazards) + len(low_hazards)
                        
                        context += f"  Total Hazards: {total_hazards}\n"
                        
                        if high_hazards:
                            context += f"  High Priority Hazards: {', '.join(high_hazards)}\n"
                        
                        if medium_hazards:
                            context += f"  Medium Priority Hazards: {', '.join(medium_hazards)}\n"
                        
                        if low_hazards:
                            context += f"  Low Priority Hazards: {', '.join(low_hazards)}\n"
                    
                    # Add recommendations
                    recommendations = best_log.get("recommendations", [])
                    if recommendations:
                        context += f"  Recommendations: {', '.join(recommendations)}\n"
                
                context += "\nBest Configuration for Each Space:\n"
                for space_id, config in optimal_configs.items():
                    if space_id != global_best_space_id:  # Skip the global best as it's already shown
                        best_log = config.get("best_log", {})
                        
                        context += f"  {config['space_name']}:\n"
                        context += f"    Best Safety Score: {best_log.get('score', 'Unknown')}\n"
                        context += f"    Date: {best_log.get('created_at', 'Unknown')}\n"
                
                context += "\n"
        
        # Generate the response
        prompt = f"""
        You are an AI assistant for a healthcare safety application.
        Your role is to help users understand safety risks in their spaces (rooms) and provide recommendations to improve safety for patients.
        
        USER QUERY: {last_message.content}
        
        AVAILABLE CONTEXT:
        {context}
        
        IMPORTANT RESPONSE GUIDELINES:
        1. Be concise and to the point - use short sentences and bullet points where appropriate
        2. ONLY provide information that is directly supported by the data provided above
        3. DO NOT add speculative details or make assumptions beyond what is explicitly in the data
        4. DO NOT hallucinate information that isn't present in the context
        5. Keep your response brief - aim for 3-5 sentences maximum unless listing specific items
        6. Focus only on answering the user's specific question
        
        Based on ONLY the facts in the provided context, give a brief, direct response:
        """
        
        response = self.model.generate_content(prompt)
        
        # Update the conversation history
        state.conversation_history.append(Message(role="assistant", content=response.text))
        
        return state
    
    async def process_conversation(self, user_id: str, query: str, history: List[Message], db: Collection) -> str:
        """Process a conversation turn and return the agent's response."""
        # Initialize state with conversation history and user info
        conversation_history = history.copy()
        
        # Add the new user message
        conversation_history.append(Message(role="user", content=query))
        
        # Initialize state
        state = AgentState(
            conversation_history=conversation_history,
            user_id=user_id
        )
        
        # Run the agent graph
        try:
            # Process understanding
            state = self._understand_query(state)
            
            # Process patient retrieval
            state = await self._retrieve_patient_info(state, db)
            
            # Process space retrieval
            state = await self._retrieve_space_info(state, db)
            
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