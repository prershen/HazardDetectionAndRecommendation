from fastapi import APIRouter, HTTPException, Depends, status
from spaces.models.space_models import SpaceImageUpload, SpaceUpdate, SpaceCreate, BoundingBox, SpaceLogCreate, SpaceLogUpdate, SpaceResponse, SpaceLogResponse, HazardList
from spaces.controllers.space_controllers import SpaceController, SpaceLogController
from patients.controllers.patient_controllers import PatientController
from user.db.mongodb import get_db
from pymongo.collection import Collection
from user.utils.auth import get_current_user
from typing import List, Dict, Any, Optional
import base64
from pydantic import BaseModel
from bson.objectid import ObjectId
import os
import uuid
import boto3
from openai import OpenAI
import instructor
from io import BytesIO
from PIL import Image, ImageDraw
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import numpy as np

router = APIRouter()
patient_controller = PatientController()
space_controller = SpaceController()
space_log_controller = SpaceLogController()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "HELLO")
GEMINI_MODEL = "gemini-2.0-flash"

client = OpenAI(
  api_key=GEMINI_API_KEY,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
bucket_name = os.getenv('BUCKET_NAME')

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

client = instructor.from_openai(client)

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")


class Hazard(BaseModel):
  priority: str
  description: str
  score: float

class Response(BaseModel):
  hazards: list[Hazard]
  recommendations: list[str]

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
        if space.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to create a space for another user"
            )
            
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

@router.get("/user/{user_id}", response_model=List[SpaceResponse])
async def get_spaces_by_user(
    user_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # Only allow users to access their own spaces
    if user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access another user's spaces"
        )
        
    spaces = await space_controller.get_spaces_by_user_id(user_id, db)
    return spaces

@router.get("/{space_id}", response_model=SpaceResponse)
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

@router.put("/{space_id}", response_model=SpaceResponse)
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
@router.post("/logs/create", response_model=SpaceLogResponse)
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

def _upload_blob(file):
    """Uploads a file to the bucket."""
    file_extension = file.filename.split('.')[-1]
    file_key = f"uploads/{uuid.uuid4()}.{file_extension}"

    s3.upload_fileobj(
        file.file,
        bucket_name,
        file_key,
        ExtraArgs={"ContentType": file.content_type}
    )

    file_url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"

    print(
        f"File {file} uploaded to {file_key}."
    )
    return {"url": file_url}

def _get_hazards_and_recommendations(image, patients):
    print("Inside _get_hazards_and_recommendations")
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    base64_img = encode_image(image)
    prompt = ""
    hazard_list = HazardList()
    for patient in patients:
        if patient.patient_condition:
            prompt += "User has a condition: " + patient.patient_condition + ". "
    print("Prompt of condition: ", prompt)
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt + " Generate a list of up to 15 hazards this user might face in this room. Be specific and prioritize based on user's condition. Respond in the form of a dictionary of priority and list of objects without explanation and their respective risk score out of 1 with 0 being lowest risk and 1 being highest risk. \
                        Give in the following format\
                        High Priority: {\"A\": scoreA, \"B\": scoreB} \
                        Medium Priority: {\"C\": scoreC, \"D\": scoreD}\
                        Low Priority: {\"E\": scoreE, \"F\": scoreF} \
                        For example, if there is an image with a rug near the desk, an unstable chair with wheels, poor lighting, sharp edged table close to door, and low seating, the model should respond as follows. \
                        High Priority: {\"Rug near the desk\": 0.9, \"Unstable chair with wheels\": 0.85, \"Sharp edged table near door\": 0.95} \
                        Medium Priority: {\"low seating\": 0.6}\
                        Low Priority: {\"Poor lighting\": 0.3}"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_img}"
                    },
                    },
                ],
                },
                {
                "role": "system",
                "content": "You are a recommender system that is an expert at safe room configurations."
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Given the hazards, give recommendations to the user to rearrange the room in a way it minimizes the hazards based on user's condition with more important recommendations in the beginning."
                    },
                ]
                }
            ],
            response_model = Response
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Get hazards and recommendation failed: {str(e)}"
        )


    hazards, recommendations = response.hazards, response.recommendations

    for hazard in hazards:
        if "high" in hazard.priority.lower():
            hazard_list.high_priority.append(hazard.description)
        elif "medium" in hazard.priority.lower():
            hazard_list.medium_priority.append(hazard.description)
        else:
            hazard_list.low_priority.append(hazard.description)
    print("hazard_list: ", hazard_list)
    return hazard_list, hazards, recommendations

def _get_safety_score(hazards):
    print("Inside _get_risk_score")
    total_score = 0
    total_weight = 0
    priority_weights = {
        "high": 3,
        "medium": 2,
        "low": 1
    }
    for hazard in hazards:
        for priority in ["high", "medium", "low"]:
            if priority in hazard.priority.lower():
                total_score += hazard.score * priority_weights[priority]
                total_weight += priority_weights[priority]
    risk_score = total_score / total_weight
    print("Risk score: ", risk_score)
    return round((1 - risk_score) * 100)

def _get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

def _get_bounding_box(hazard_list, image):
    print("Inside _get_bounding_box")
    hazards = [hazard_list.high_priority + hazard_list.medium_priority + hazard_list.low_priority]
    print("hazards texts: ", hazards)
    
    bounding_box_info = []
    threshold = 0.25

    try:
        print("Sending the inputs through processor")
        inputs = processor(text=hazards, images=image, return_tensors="pt")
        with torch.no_grad():
            print("Forward pass")
            outputs = model(**inputs)

        print("Get preprocessed image")
        unnormalized_image = _get_preprocessed_image(inputs.pixel_values)

        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        print("Sending post_process_grounded_object_detection with threshold ", threshold)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=hazards
        )
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = hazards[i]
        visualized_image = unnormalized_image.copy()
        draw = ImageDraw.Draw(visualized_image)
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            bb = BoundingBox(box, score, label)
            bounding_box_info.append(bb)
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=text[label])
        return bounding_box_info, visualized_image
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Get Bounding Box failed: {str(e)}"
        )
            
    
@router.post("/logs/upload-image", response_model=SpaceLogResponse)
async def upload_image(
    payload: SpaceImageUpload,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    print("Inside upload_image with space_id ", payload.space_id)
    try:
        # First verify that the space exists and belongs to the current user
        space = await space_controller.get_space_by_id(payload.space_id, db)
        
        if not space:
            raise HTTPException(status_code=404, detail="Space not found")
        
        if space.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to add logs to this space"
            )
        
        # Read and encode the image
        orig_image_url = _upload_blob(payload.image)
        contents = await payload.image.read()
        img = Image.open(BytesIO(contents))

        print("Getting patient info for ", space.patient_ids)
        patients = [await patient_controller.get_patient_by_id(patient_id, db) for patient_id in space.patient_ids]
        
        print("Getting hazards and recommendations")
        hazard_list, hazards, recommendations = _get_hazards_and_recommendations(img, patients)
        
        score = _get_safety_score(hazards)
        print("Safety Score: ", score)

        bounding_box_info = []

        bounding_box_info, image_with_bb = _get_bounding_box(hazard_list, img)
        print("Bounding box: ", bounding_box_info)

        bb_image_url = -_upload_blob(image_with_bb)

        # Create the space log
        space_log_data = SpaceLogCreate(
            space_id=payload.space_id,
            image_data=orig_image_url,
            image_bb=bb_image_url,
            score=score,
            bounding_box_info=bounding_box_info,
            recommendations=recommendations,
            hazard_list=hazard_list,
            comments=payload.comments,
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

@router.get("/logs/{log_id}", response_model=SpaceLogResponse)
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

@router.get("/logs/space/{space_id}", response_model=List[SpaceLogResponse])
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

@router.put("/logs/{log_id}", response_model=SpaceLogResponse)
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

@router.post("/{space_id}/patients", response_model=SpaceResponse)
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

@router.get("/{space_id}/patients")
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