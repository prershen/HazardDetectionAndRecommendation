from typing import Optional, List
from pydantic import BaseModel, Field, validator
from bson import ObjectId


class Patient(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    patient_name: str
    user_id: str
    patient_condition: str
    patient_relationship: Optional[str] = None
    patient_age: int
    medical_history: Optional[str] = None
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, v):
        return str(v) if isinstance(v, ObjectId) else v

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class PatientCreate(BaseModel):
    patient_name: str
    patient_condition: str
    patient_relationship: Optional[str] = None
    patient_age: int
    medical_history: Optional[str] = None


class PatientUpdate(BaseModel):
    patient_name: Optional[str] = None
    patient_condition: Optional[str] = None
    patient_relationship: Optional[str] = None
    patient_age: Optional[int] = None
    medical_history: Optional[str] = None


class PatientResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    patient_name: str
    user_id: str
    patient_condition: str
    patient_relationship: Optional[str] = None
    patient_age: int
    medical_history: Optional[str] = None
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, v):
        return str(v) if isinstance(v, ObjectId) else v

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}