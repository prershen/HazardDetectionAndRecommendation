from typing import Optional, List
from pydantic import BaseModel, Field, validator
from bson import ObjectId


class Patient(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    patient_name: str
    user_id: str
    patient_condition: str
    patient_age: int
    medical_history: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value


class PatientCreate(BaseModel):
    patient_name: str
    patient_condition: str
    patient_age: int
    medical_history: Optional[str] = None


class PatientUpdate(BaseModel):
    patient_name: Optional[str] = None
    patient_condition: Optional[str] = None
    patient_age: Optional[int] = None
    medical_history: Optional[str] = None


class PatientResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    patient_name: str
    user_id: str
    patient_condition: str
    patient_age: int
    medical_history: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True
    
    @validator("id", pre=True, always=True)
    def convert_objectid(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value 