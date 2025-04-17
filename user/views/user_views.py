from fastapi import APIRouter, HTTPException, Depends, status
from user.models.user_models import User, UserUpdate, UserCreate, UserLogin, Token, UserResponse
from user.controllers.user_controllers import UserController
from user.db.mongodb import get_db
from pymongo.collection import Collection
from datetime import timedelta
from user.utils.auth import create_access_token, get_current_user
from core.config import settings

router = APIRouter()
user_controller = UserController()

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Collection = Depends(get_db)):
    try:
        # Add validation for email format
        if '@' not in user.email:
            raise ValueError("Invalid email format")
            
        # Log the incoming data (for debugging)
        print(f"Registering user: {user.username}, Email: {user.email}")
        
        created_user = await user_controller.create_user(user, db)
        if not created_user:
            raise HTTPException(status_code=400, detail="User creation failed")
        return created_user
    except ValueError as e:
        print(f"Registration error (ValueError): {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Registration error (Exception): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login_user(login_data: UserLogin, db: Collection = Depends(get_db)):
    user = await user_controller.authenticate_user(login_data.username, login_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.jwt_expiration_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    print(f"Current user ID from token: {current_user_id}")
    # Add more debugging
    if not current_user_id:
        print("Warning: current_user_id is empty")
        raise HTTPException(status_code=401, detail="Authentication required")
        
    user = await user_controller.get_user_by_id(current_user_id, db)
    
    # Add more debugging
    if not user:
        print(f"No user found in database with ID: {current_user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    print(f"Found user: {user}")
    return user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,  
    db: Collection = Depends(get_db), 
    current_user_id: str = Depends(get_current_user)
):
    print(f"get_user: Accessing user_id {user_id} by authenticated user {current_user_id}")
    # For security, users can only view their own profile unless admin role is implemented
    if user_id != current_user_id:
        print(f"Access denied: {current_user_id} tried to access {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's data"
        )
        
    user = await user_controller.get_user_by_id(user_id, db)
    if not user:
        print(f"User not found: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    print(f"User found: {user}")
    return user

@router.put("/{user_id}/update", response_model=UserResponse)
async def update_user(
    user_id: str, 
    user: UserUpdate, 
    db: Collection = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    print(f"update_user: Updating user_id {user_id} by authenticated user {current_user_id}")
    print(f"Update data: {user.dict(exclude_unset=True)}")
    
    # For security, users can only update their own profile unless admin role is implemented
    if user_id != current_user_id:
        print(f"Update denied: {current_user_id} tried to update {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's data"
        )
        
    try:
        updated_user = await user_controller.update_user(user_id, user, db)
        if not updated_user:
            print(f"Update failed: User not found {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        print(f"User updated successfully: {updated_user}")
        return updated_user
    except ValueError as e:
        print(f"Update error (ValueError): {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Update error (Exception): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Update failed: {str(e)}"
        )