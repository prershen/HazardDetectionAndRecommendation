from pymongo.errors import DuplicateKeyError
from patients.models.patient_models import Patient, PatientUpdate, PatientCreate
from pymongo.collection import Collection
from bson.objectid import ObjectId
from typing import Optional, List


class PatientController:
    async def create_patient(self, user_id: str, patient_data: PatientCreate, db: Collection) -> Optional[Patient]:
        patients_collection = db["patients"]
        
        try:
            # Create patient dictionary
            patient_dict = patient_data.dict()
            patient_dict["user_id"] = user_id  # Add the user_id from the authenticated user
            
            # Insert the patient data into the collection
            result = patients_collection.insert_one(patient_dict)
            
            # Fetch the inserted document
            created_patient = patients_collection.find_one(
                {"_id": result.inserted_id})
                
            if created_patient is None:
                raise ValueError("Patient was not created successfully.")
                
            return Patient(**created_patient)
            
        except Exception as e:
            print(f"Error in create_patient: {str(e)}")
            raise ValueError(f"Patient creation failed: {str(e)}")
    
    async def get_patient_by_id(self, patient_id: str, db: Collection) -> Optional[Patient]:
        patients_collection = db["patients"]
        patient_data = patients_collection.find_one({"_id": ObjectId(patient_id)})
        
        if patient_data:
            return Patient(**patient_data)
        return None
    
    async def get_patients_by_user_id(self, user_id: str, db: Collection) -> List[Patient]:
        patients_collection = db["patients"]
        patients = []
        
        cursor = patients_collection.find({"user_id": user_id})
        for patient_data in cursor:
            patients.append(Patient(**patient_data))
            
        return patients
    
    async def update_patient(self, patient_id: str, patient_data: PatientUpdate, db: Collection) -> Optional[Patient]:
        patients_collection = db["patients"]
        patient_dict = patient_data.dict(exclude_unset=True)
        
        try:
            updated_patient = patients_collection.find_one_and_update(
                {"_id": ObjectId(patient_id)},
                {"$set": patient_dict},
                return_document=True
            )
            
            if updated_patient:
                return Patient(**updated_patient)
            return None
            
        except Exception as e:
            print(f"Error in update_patient: {str(e)}")
            raise ValueError(f"Patient update failed: {str(e)}")
    
    async def delete_patient(self, patient_id: str, db: Collection) -> bool:
        patients_collection = db["patients"]
        
        try:
            result = patients_collection.delete_one({"_id": ObjectId(patient_id)})
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"Error in delete_patient: {str(e)}")
            raise ValueError(f"Patient deletion failed: {str(e)}") 