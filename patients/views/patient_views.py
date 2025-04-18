from fastapi import APIRouter, HTTPException, Depends, status
from patients.models.patient_models import Patient, PatientUpdate, PatientCreate, PatientResponse
from patients.controllers.patient_controllers import PatientController
from user.db.mongodb import get_db
from pymongo.collection import Collection
from user.utils.auth import get_current_user
from typing import List

router = APIRouter()
patient_controller = PatientController()


@router.post("/", response_model=PatientResponse, response_model_by_alias=True)
async def create_patient(
    patient: PatientCreate, 
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        created_patient = await patient_controller.create_patient(current_user_id, patient, db)
        if not created_patient:
            raise HTTPException(status_code=400, detail="Patient creation failed")
        return created_patient
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Patient creation failed: {str(e)}"
        )


@router.get("/", response_model=List[PatientResponse], response_model_by_alias=True)
async def get_user_patients(
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        patients = await patient_controller.get_patients_by_user_id(current_user_id, db)
        return patients
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve patients: {str(e)}"
        )


@router.get("/{patient_id}", response_model=PatientResponse, response_model_by_alias=True)
async def get_patient(
    patient_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    patient = await patient_controller.get_patient_by_id(patient_id, db)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Security check: ensure the patient belongs to the current user
    if patient.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this patient's data"
        )
    
    return patient


@router.put("/{patient_id}", response_model=PatientResponse, response_model_by_alias=True)
async def update_patient(
    patient_id: str,
    patient: PatientUpdate,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First verify the patient exists and belongs to the current user
    existing_patient = await patient_controller.get_patient_by_id(patient_id, db)
    
    if not existing_patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if existing_patient.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this patient's data"
        )
    
    try:
        updated_patient = await patient_controller.update_patient(patient_id, patient, db)
        if not updated_patient:
            raise HTTPException(status_code=404, detail="Patient not found or update failed")
        return updated_patient
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Update failed: {str(e)}"
        )


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: str,
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # First verify the patient exists and belongs to the current user
    existing_patient = await patient_controller.get_patient_by_id(patient_id, db)
    
    if not existing_patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if existing_patient.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this patient's data"
        )
    
    try:
        success = await patient_controller.delete_patient(patient_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Patient not found or deletion failed")
        return None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deletion failed: {str(e)}"
        ) 