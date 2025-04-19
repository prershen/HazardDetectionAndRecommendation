from fastapi import APIRouter, HTTPException, Depends, status, File, UploadFile, Form
from patients.models.patient_models import PatientResponse

from spaces.models.space_models import SpaceImageUpload, SpaceUpdate, SpaceCreate, BoundingBox, SpaceLogCreate, SpaceLogUpdate, SpaceResponse, SpaceLogResponse, HazardList, SpaceLog, Space
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
from dotenv import load_dotenv

load_dotenv()

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
  walking_path_score: float

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
        print("Creating space: ", space)
        # Ensure the user creating the space is the same as the user_id
        space.user_id = current_user_id
        created_space = await space_controller.create_space(space, db)
        if not created_space:
            raise HTTPException(status_code=400, detail="Space creation failed")
        return created_space
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Space creation failed: ", str(e))
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

def _upload_blob(file_name):
    """Uploads a file to the bucket."""
    # Check if file is a string path or an UploadFile object
    if isinstance(file_name, str):
        file_extension = file_name.split('.')[-1]
    else:
        # Handle UploadFile object
        file_extension = file_name.filename.split('.')[-1]
    file_key = f"{uuid.uuid4()}.{file_extension}"
    # file_key = os.path.basename(file_name)

    print(f"Sending request with {file_name} to {bucket_name} as {file_key}")
    try:
        if file_extension in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "ico", "webp"]:
            print("Uploaading image to  s3")
            s3.upload_file(
            file_name,
            bucket_name,
            file_key)
        else:
            print("Uploaading file to  s3")
            s3.upload_fileobj(
                file_name,
                bucket_name,
                file_key,
                ExtraArgs={"ContentType": file_extension}
            )
        print("File uploaded to s3", file_key)
    except Exception as e:
        print("Error in uploading file to s3", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"s3 image upload failed: {str(e)}"
        )

    file_url = f"https://{bucket_name}.s3.amazonaws.com/{file_key}"

    print(
        f"File {file_name} uploaded to {file_key}."
    )
    return file_url

def _get_hazards_and_recommendations(image, patients):
    print("Inside _get_hazards_and_recommendations")
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    base64_img = encode_image(image)
    print("Encoded image: ", type(base64_img))
    prompt = ""
    hazard_list = HazardList()
    for patient in patients:
        if patient.patient_condition:
            prompt += "User has a condition: " + patient.patient_condition + ". "
    print("Prompt of condition: ", prompt)
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "user", "content": [
                    {
                    "type": "text",
                    "text": prompt + " Generate a list of up to 15 hazards this user might face in this room. Be specific and prioritize based on user's condition. Respond in the form of a dictionary of priority and list of objects without explanation and their respective risk score out of 1 with 0 being lowest risk and 1 being highest risk. \
                        Also, give a walking path score (0-1) for how easily the user can navigate through the room. \
                        Give in the following format\
                        High Priority: {\"A\": scoreA, \"B\": scoreB} \
                        Medium Priority: {\"C\": scoreC, \"D\": scoreD}\
                        Low Priority: {\"E\": scoreE, \"F\": scoreF} \
                        Walking Path: {\"G\": scoreG} \
                        For example, if there is an image with a rug near the desk, an unstable chair with wheels, poor lighting, sharp edged table close to door, and low seating, the model should respond as follows. \
                        High Priority: {\"Rug near the desk\": 0.9, \"Unstable chair with wheels\": 0.85, \"Sharp edged table near door\": 0.95} \
                        Medium Priority: {\"low seating\": 0.6}\
                        Low Priority: {\"Poor lighting\": 0.3} \
                        Walking Path: {\"G\": 0.8}"
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
        "response_model": Response
    }
    try:
        # response = client.chat.completions.create(
        #     model=GEMINI_MODEL,
        #     messages=[
        #         {
        #         "role": "user",
        #         "content": [
        #             {
        #             "type": "text",
        #             "text": prompt + " Generate a list of up to 15 hazards this user might face in this room. Be specific and prioritize based on user's condition. Respond in the form of a dictionary of priority and list of objects without explanation and their respective risk score out of 1 with 0 being lowest risk and 1 being highest risk. \
        #                 Give in the following format\
        #                 High Priority: {\"A\": scoreA, \"B\": scoreB} \
        #                 Medium Priority: {\"C\": scoreC, \"D\": scoreD}\
        #                 Low Priority: {\"E\": scoreE, \"F\": scoreF} \
        #                 For example, if there is an image with a rug near the desk, an unstable chair with wheels, poor lighting, sharp edged table close to door, and low seating, the model should respond as follows. \
        #                 High Priority: {\"Rug near the desk\": 0.9, \"Unstable chair with wheels\": 0.85, \"Sharp edged table near door\": 0.95} \
        #                 Medium Priority: {\"low seating\": 0.6}\
        #                 Low Priority: {\"Poor lighting\": 0.3}"
        #             },
        #             {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url":  f"data:image/jpeg;base64,{base64_img}"
        #             },
        #             },
        #         ],
        #         },
        #         {
        #         "role": "system",
        #         "content": "You are a recommender system that is an expert at safe room configurations."
        #         },
        #         {
        #         "role": "user",
        #         "content": [
        #             {
        #             "type": "text",
        #             "text": "Given the hazards, give recommendations to the user to rearrange the room in a way it minimizes the hazards based on user's condition with more important recommendations in the beginning."
        #             },
        #         ]
        #         }
        #     ],
        #     response_model = Response
        # )
        response = client.chat.completions.create(**payload)
        if not response.hazards or not response.recommendations:
            response = client.chat.completions.create(**payload)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Get hazards and recommendation failed: {str(e)}"
        )


    hazards, recommendations, walking_path_score = response.hazards, response.recommendations, response.walking_path_score
    print("Hazards: ", hazards)
    print("Recommendations: ", recommendations)
    print("Walking path score: ", walking_path_score)
    for hazard in hazards:
        if "high" in hazard.priority.lower():
            hazard_list.high_priority.append(hazard.description)
        elif "medium" in hazard.priority.lower():
            hazard_list.medium_priority.append(hazard.description)
        else:
            hazard_list.low_priority.append(hazard.description)
    print("hazard_list: ", hazard_list)
    return hazard_list, hazards, recommendations, walking_path_score

def _get_safety_score(hazards, walking_path_score):
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
    risk_score = total_score / total_weight if total_weight > 0 else 1
    risk_score = round(risk_score * 100)
    print("Risk score: ", risk_score)
    return 100 - (risk_score - risk_score * walking_path_score)

def _get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

def compute_iou(box1, box2):
    """Compute IoU between two boxes: (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def remove_duplicates(bbs):
    """Remove fully overlapping boxes (IoU = 1) with the same label"""
    filtered = []
    for i, (box1, score1, label1) in enumerate(bbs):
        is_duplicate = False
        for j, (box2, score2, label2) in enumerate(filtered):
            if label1 == label2 and compute_iou(box1.tolist(), box2.tolist()) == 1.0:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append((box1, score1, label1))
    return filtered

def _get_bounding_box(hazard_list, image):
    print("Inside _get_bounding_box")
    hazards = [hazard_list.high_priority + hazard_list.medium_priority + hazard_list.low_priority]
    print("hazards texts: ", hazards)
    
    bounding_box_info = []
    threshold = 0.15

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
        sorted_bb = sorted(list(zip(boxes, scores, labels)), key=lambda x:x[1], reverse=True)
        if len(sorted_bb) == 0 and (hazard_list.high_priority or hazard_list.medium_priority):
            raise HTTPException(
                status_code=500,
                detail=f"No bounding boxes found for {text}"
            )
        high_priority_sorted_bb = []
        medium_priority_sorted_bb = []
        low_priority_sorted_bb = []
        for box, score, label in zip(boxes, scores, labels):
            if text[label] in hazard_list.high_priority:
                print("label in high priority: ", label)
                high_priority_sorted_bb.append((box, score, label))
            elif text[label] in hazard_list.medium_priority:
                print("label in medium priority: ", label)
                medium_priority_sorted_bb.append((box, score, label))
            else:
                print("label in low priority: ", label)
                low_priority_sorted_bb.append((box, score, label))
        iter_list = remove_duplicates(high_priority_sorted_bb + medium_priority_sorted_bb)
        print("iter_list: ", iter_list)
        for box, score, label in iter_list:
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            bb = BoundingBox(box=box, score=score, label=text[label])
            bounding_box_info.append(bb)
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=text[label])
        return bounding_box_info, visualized_image
            
    except Exception as e:
        print("Error in _get_bounding_box", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Get Bounding Box failed: {str(e)}"
        )
            
    
@router.post("/logs/upload-image", response_model=SpaceLogResponse)
async def upload_image(
    space_id: str = Form(...),
    image: UploadFile = File(...),
    comments: Optional[str] = Form(None),
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    print("Inside upload_image with space_id ", space_id)
    image_path = "/home/prathik-somanath/git/responsible/backend-server/temp_image.png"
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
        print("Contents type: ", type(contents))
        img = Image.open(BytesIO(contents))
        print("Image type: ", type(img))
        img.save(image_path, "PNG")
        orig_image_url = _upload_blob(image_path)

        print("Getting patient info for ", space.patient_ids)
        patients = [await patient_controller.get_patient_by_id(patient_id, db) for patient_id in space.patient_ids]
        
        print("Getting hazards and recommendations")
        hazard_list, hazards, recommendations, walking_path_score = _get_hazards_and_recommendations(image_path, patients)
        
        score = _get_safety_score(hazards, walking_path_score)
        print("Safety Score: ", score)

        bounding_box_info = []

        bounding_box_info, image_with_bb = _get_bounding_box(hazard_list, img)
        print("Bounding box: ", bounding_box_info)

        image_with_bb.save(image_path, "PNG")
        bb_image_url = _upload_blob(image_path)

        # Create the space log
        space_log_data = SpaceLogCreate(
            space_id=space_id,
            image_data=orig_image_url,
            image_bb=bb_image_url,
            score=score,
            bounding_box_info=bounding_box_info,
            recommendations=recommendations[:10],
            hazard_list=hazard_list,
            comments=None,
        )
        
        created_log = await space_log_controller.create_space_log(space_log_data, db)
        if not created_log:
            print(created_log)
            raise HTTPException(status_code=400, detail="Space log creation failed")
        return created_log
    except ValueError as e:
        print("Error in upload_image", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Error in upload_image", str(e))
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