from pymongo.errors import DuplicateKeyError
from user.models.user_models import User, UserUpdate, UserCreate, UserResponse
from pymongo.collection import Collection
from bson.objectid import ObjectId
from typing import Optional
from user.utils.auth import get_password_hash, verify_password


class UserController:
    async def create_user(self, user_data: UserCreate, db: Collection) -> Optional[User]:
        users_collection = db["users"]
        
        try:
            # Check if username already exists
            existing_user = users_collection.find_one({"username": user_data.username})
            if existing_user:
                raise ValueError("Username already exists")
            
            # Check if email already exists
            existing_email = users_collection.find_one({"email": user_data.email})
            if existing_email:
                raise ValueError("Email already exists")
        
            # Create user dictionary and hash the password
            user_dict = user_data.dict(exclude={"password"})
            user_dict["hashed_password"] = get_password_hash(user_data.password)

            # Insert the user data into the collection
            result = users_collection.insert_one(user_dict)

            # Fetch the inserted document using the correct ObjectId query
            created_user = users_collection.find_one(
                {"_id": result.inserted_id})

            if created_user is None:
                raise ValueError("User was not created successfully.")

            return User(**created_user)

        except DuplicateKeyError as e:
            # Check which field caused the error
            error_msg = str(e)
            if "username" in error_msg:
                raise ValueError("Username already exists.")
            elif "email" in error_msg:
                raise ValueError("Email already exists.")
            else:
                raise ValueError(f"Duplicate key error: {error_msg}")
        except Exception as e:
            print(f"Error in create_user: {str(e)}")
            raise ValueError(f"User creation failed: {str(e)}")

    async def get_user_by_id(self, user_id: str, db: Collection) -> Optional[User]:
        users_collection = db["users"]
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})

        if user_data:
            return UserResponse(**user_data)
        return None

    async def get_user_by_username(self, username: str, db: Collection) -> Optional[User]:
        users_collection = db["users"]
        user_data = users_collection.find_one({"username": username})

        if user_data:
            return UserResponse(**user_data)
        return None

    async def authenticate_user(self, username: str, password: str, db: Collection) -> Optional[User]:
        user = await self.get_user_by_username(username, db)
        
        if not user or not user.hashed_password:
            return None
            
        if not verify_password(password, user.hashed_password):
            return None
            
        return user

    async def update_user(self, user_id: str, user_data: UserUpdate, db: Collection) -> Optional[User]:
        users_collection = db["users"]
        user_dict = user_data.dict(exclude_unset=True)

        # If there's a password in the update, hash it
        if "password" in user_dict:
            user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))

        try:
            updated_user = users_collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {"$set": user_dict},
                return_document=True
            )

            if updated_user:
                updated_user["_id"] = str(updated_user["_id"])
                return UserResponse(**updated_user)
            return None
        except DuplicateKeyError as e:
            # Check which field caused the error
            if "username" in str(e):
                raise ValueError("Username already exists.")
            elif "email" in str(e):
                raise ValueError("Email already exists.")
            else:
                raise ValueError("Username or email already exists.")
