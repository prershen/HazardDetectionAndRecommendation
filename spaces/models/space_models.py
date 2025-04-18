from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from bson import ObjectId


class Space(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    space_name: str
    user_id: str
    description: Optional[str] = None
    patient_ids: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value


class SpaceCreate(BaseModel):
    space_name: str
    user_id: str
    description: Optional[str] = None
    patient_ids: Optional[List[str]] = None


class SpaceUpdate(BaseModel):
    space_name: Optional[str] = None
    description: Optional[str] = None
    patient_ids: Optional[List[str]] = None
    
    class Config:
        allow_population_by_field_name = True


class HazardList(BaseModel):
    high_priority: List[str] = []
    medium_priority: List[str] = []
    low_priority: List[str] = []


class SpaceLog(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    space_id: str
    image_data: str  # Base64 encoded image
    score: Optional[float] = None
    bounding_box_info: Dict[str, Any] = {}
    recommendations: Optional[List[str]] = []
    hazard_list: HazardList = Field(default_factory=HazardList)
    comments: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value


class SpaceLogCreate(BaseModel):
    space_id: str
    image_data: str
    score: Optional[float] = None
    bounding_box_info: Dict[str, Any] = {}
    recommendations: Optional[List[str]] = []
    hazard_list: Optional[HazardList] = None
    comments: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SpaceLogUpdate(BaseModel):
    image_data: Optional[str] = None
    score: Optional[float] = None
    bounding_box_info: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    hazard_list: Optional[HazardList] = None
    comments: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        allow_population_by_field_name = True


class SpaceResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    space_name: str
    user_id: str
    description: Optional[str] = None
    patient_ids: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, v):
        return str(v) if isinstance(v, ObjectId) else v


class SpaceLogResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    space_id: str
    image_data: Optional[str] = None
    score: Optional[float] = None
    bounding_box_info: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    hazard_list: Optional[HazardList] = None
    comments: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, v):
        return str(v) if isinstance(v, ObjectId) else v