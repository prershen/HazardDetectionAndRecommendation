from fastapi import APIRouter, HTTPException, Depends, status, File, UploadFile, Form
from spaces.models.space_models import Space, SpaceUpdate, SpaceCreate, SpaceLog, SpaceLogCreate, SpaceLogUpdate, SpaceResponse, SpaceLogResponse, HazardList
from patients.models.patient_models import PatientResponse
from spaces.controllers.space_controllers import SpaceController, SpaceLogController
from user.db.mongodb import get_db
from pymongo.collection import Collection
from user.utils.auth import get_current_user
from typing import List, Dict, Any, Optional
import base64
import json
from datetime import datetime
from pydantic import BaseModel
from bson.objectid import ObjectId
from fastapi.responses import JSONResponse

router = APIRouter()
space_controller = SpaceController()
space_log_controller = SpaceLogController()

# New models for patient-space operations
class PatientToSpaceRequest(BaseModel):
    patient_id: str

@router.post("/create", response_model=SpaceResponse)
async def create_space(
    space: SpaceCreate, 
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        # Ensure the user creating the space is the same as the user_id
        space.user_id = current_user_id
        created_space = await space_controller.create_space(space, db)
        if not created_space:
            raise HTTPException(status_code=400, detail="Space creation failed")
        return created_space
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Space creation failed: {str(e)}"
        )

@router.get("/user", response_model=List[SpaceResponse], response_model_by_alias=True)
async def get_spaces_by_user(
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    spaces = await space_controller.get_spaces_by_user_id(current_user_id, db)
    # return JSONResponse(content=spaces.dict(by_alias=True))
    return spaces

@router.get("/{space_id}", response_model=SpaceResponse, response_model_by_alias=True)
async def get_space(
    space_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Only allow users to access their own spaces
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this space"
        )
    return space

@router.put("/{space_id}", response_model=SpaceResponse, response_model_by_alias=True)
async def update_space(
    space_id: str,
    space_update: SpaceUpdate,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the space to check ownership
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Only allow users to update their own spaces
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this space"
        )
    
    try:
        updated_space = await space_controller.update_space(space_id, space_update, db)
        if not updated_space:
            raise HTTPException(status_code=404, detail="Space not found")
        return updated_space
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Space update failed: {str(e)}"
        )

@router.delete("/{space_id}")
async def delete_space(
    space_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the space to check ownership
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Only allow users to delete their own spaces
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this space"
        )
    
    result = await space_controller.delete_space(space_id, db)
    
    if not result:
        raise HTTPException(status_code=404, detail="Space not found or deletion failed")
    
    return {"message": "Space and associated logs deleted successfully"}

# Space Log Endpoints
@router.post("/logs/create", response_model=SpaceLogResponse, response_model_by_alias=True)
async def create_space_log(
    space_log: SpaceLogCreate,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        # First verify that the space exists and belongs to the current user
        space = await space_controller.get_space_by_id(space_log.space_id, db)
        
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")
        
        if space.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to add logs to this space"
            )
            
        created_log = await space_log_controller.create_space_log(space_log, db)
        if not created_log:
            raise HTTPException(status_code=400, detail="Space log creation failed")
        return created_log
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Space log creation failed: {str(e)}"
        )

@router.post("/logs/upload-image", response_model=SpaceLogResponse, response_model_by_alias=True)
async def upload_image(
    space_id: str = Form(...),
    image: UploadFile = File(...),
    score: Optional[float] = Form(None),
    bounding_box_info: str = Form("{}"),
    recommendations: str = Form("[]"),
    hazard_list: str = Form("{}"),
    comments: Optional[str] = Form(None),
    metadata: str = Form("{}"),
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        # First verify that the space exists and belongs to the current user
        space = await space_controller.get_space_by_id(space_id, db)
        
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")
        
        if space.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to add logs to this space"
            )
        
        # Read and encode the image
        contents = await image.read()
        encoded_image = base64.b64encode(contents).decode("utf-8")
        
        # Parse JSON strings to Python objects
        try:
            bounding_box_dict = json.loads(bounding_box_info)
            recommendations_list = json.loads(recommendations)
            hazard_list_dict = json.loads(hazard_list)
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        
        # Create hazard list object
        hazard_list_obj = HazardList(
            high_priority=hazard_list_dict.get("high_priority", []),
            medium_priority=hazard_list_dict.get("medium_priority", []),
            low_priority=hazard_list_dict.get("low_priority", [])
        )
        
        # Create the space log
        space_log_data = SpaceLogCreate(
            space_id=space_id,
            image_data=encoded_image,
            score=score,
            bounding_box_info=bounding_box_dict,
            recommendations=recommendations_list,
            hazard_list=hazard_list_obj,
            comments=comments,
            metadata=metadata_dict
        )
        
        created_log = await space_log_controller.create_space_log(space_log_data, db)
        if not created_log:
            raise HTTPException(status_code=400, detail="Space log creation failed")
        return created_log
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Space log creation failed: {str(e)}"
        )

@router.get("/logs/{log_id}", response_model=SpaceLogResponse, response_model_by_alias=True)
async def get_space_log(
    log_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    log = await space_log_controller.get_space_log_by_id(log_id, db)
    
    if not log:
        raise HTTPException(status_code=404, detail="Space log not found")
    
    # Verify that the log belongs to a space owned by the current user
    space = await space_controller.get_space_by_id(log.space_id, db)
    
    if not space or space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this log"
        )
    return log

@router.get("/logs/space/{space_id}", response_model=List[SpaceLogResponse], response_model_by_alias=True)
async def get_logs_by_space(
    space_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First verify that the space exists and belongs to the current user
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access logs for this space"
        )
        
    logs = await space_log_controller.get_logs_by_space_id(space_id, db)
    return logs

@router.put("/logs/{log_id}", response_model=SpaceLogResponse, response_model_by_alias=True)
async def update_space_log(
    log_id: str,
    log_update: SpaceLogUpdate,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the log to check ownership
    log = await space_log_controller.get_space_log_by_id(log_id, db)
    
    if not log:
        raise HTTPException(status_code=404, detail="Space log not found")
    
    # Verify that the log belongs to a space owned by the current user
    space = await space_controller.get_space_by_id(log.space_id, db)
    
    if not space or space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this log"
        )
    
    try:
        updated_log = await space_log_controller.update_space_log(log_id, log_update, db)
        if not updated_log:
            raise HTTPException(status_code=404, detail="Space log not found")
        return updated_log
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Space log update failed: {str(e)}"
        )

@router.delete("/logs/{log_id}")
async def delete_space_log(
    log_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the log to check ownership
    log = await space_log_controller.get_space_log_by_id(log_id, db)
    
    if not log:
        raise HTTPException(status_code=404, detail="Space log not found")
    
    # Verify that the log belongs to a space owned by the current user
    space = await space_controller.get_space_by_id(log.space_id, db)
    
    if not space or space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this log"
        )
    
    result = await space_log_controller.delete_space_log(log_id, db)
    
    if not result:
        raise HTTPException(status_code=404, detail="Space log not found or deletion failed")
    
    return {"message": "Space log deleted successfully"}

@router.post("/{space_id}/patients", response_model=SpaceResponse, response_model_by_alias=True)
async def add_patient_to_space(
    space_id: str,
    request: PatientToSpaceRequest,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the space to check ownership
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Only allow users to update their own spaces
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this space"
        )
    
    # Check if the patient exists in the database
    patients_collection = db["patients"]
    patient = patients_collection.find_one({"_id": ObjectId(request.patient_id)})
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check if the patient belongs to the current user
    if patient.get("user_id") != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to add this patient to the space"
        )
    
    try:
        updated_space = await space_controller.add_patient_to_space(space_id, request.patient_id, db)
        if not updated_space:
            raise HTTPException(status_code=404, detail="Failed to add patient to space")
        return updated_space
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add patient to space: {str(e)}"
        )

@router.delete("/{space_id}/patients/{patient_id}", response_model=SpaceResponse)
async def remove_patient_from_space(
    space_id: str,
    patient_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the space to check ownership
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Only allow users to update their own spaces
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this space"
        )
    
    try:
        updated_space = await space_controller.remove_patient_from_space(space_id, patient_id, db)
        if not updated_space:
            raise HTTPException(status_code=404, detail="Failed to remove patient from space")
        return updated_space
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove patient from space: {str(e)}"
        )

@router.get("/{space_id}/patients", response_model=List[PatientResponse], response_model_by_alias=True)
async def get_patients_in_space(
    space_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First get the space to check ownership
    space = await space_controller.get_space_by_id(space_id, db)
    
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Only allow users to view their own spaces
    if space.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view patients in this space"
        )
    
    if not space.patient_ids:
        return []
    
    # Fetch patient details
    patients_collection = db["patients"]
    patients = list(patients_collection.find({"_id": {"$in": [ObjectId(pid) for pid in space.patient_ids]}}))
    
    # Convert ObjectId to string for each patient
    for patient in patients:
        if "_id" in patient:
            patient["_id"] = str(patient["_id"])
    
    return patients 