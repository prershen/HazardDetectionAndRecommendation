from typing import Optional
from pydantic import BaseModel, Field, validator, EmailStr
from bson import ObjectId


class User(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    emergency_contact: Optional[str] = None
    hashed_password: Optional[str] = None

    @validator("id", pre=True, always=True)
    def convert_objectid(cls, v):
        return str(v) if isinstance(v, ObjectId) else v

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    

class UserUpdate(BaseModel):
    username: Optional[str]
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    phone_number: Optional[str]
    emergency_contact: Optional[str] = None
    hashed_password: Optional[str]

    class Config:
        allow_population_by_field_name = True


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    emergency_contact: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    emergency_contact: Optional[str] = None
    hashed_password: Optional[str] = None

    @validator("id", pre=True, always=True)
    def convert_objectid(cls, v):
        return str(v) if isinstance(v, ObjectId) else v

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}