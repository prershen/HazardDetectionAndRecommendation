from pymongo.errors import DuplicateKeyError
from spaces.models.space_models import Space, SpaceUpdate, SpaceCreate, SpaceLog, SpaceLogCreate, SpaceLogUpdate
from pymongo.collection import Collection
from bson.objectid import ObjectId
from typing import Optional, List
from datetime import datetime


class SpaceController:
    async def create_space(self, space_data: SpaceCreate, db: Collection) -> Optional[Space]:
        spaces_collection = db["spaces"]
        
        try:
            # Create space dictionary
            space_dict = space_data.dict()
            
            # Set default empty list for patient_ids if not provided
            if space_dict.get("patient_ids") is None:
                space_dict["patient_ids"] = []
                
            space_dict["created_at"] = datetime.now()
            space_dict["updated_at"] = datetime.now()

            # Insert the space data into the collection
            result = spaces_collection.insert_one(space_dict)

            # Fetch the inserted document
            created_space = spaces_collection.find_one(
                {"_id": result.inserted_id})

            if created_space is None:
                raise ValueError("Space was not created successfully.")

            return Space(**created_space)

        except Exception as e:
            print(f"Error in create_space: {str(e)}")
            raise ValueError(f"Space creation failed: {str(e)}")

    async def get_space_by_id(self, space_id: str, db: Collection) -> Optional[Space]:
        print("Inside get_space_by_id")
        spaces_collection = db["spaces"]
        space_data = spaces_collection.find_one({"_id": ObjectId(space_id)}, sort=[("created_at", -1)])

        if space_data:
            return Space(**space_data)
        return None

    async def get_spaces_by_user_id(self, user_id: str, db: Collection) -> List[Space]:
        spaces_collection = db["spaces"]
        spaces_data = spaces_collection.find({"user_id": user_id})

        spaces = []
        for space_data in spaces_data:
            spaces.append(Space(**space_data))
        
        return spaces

    async def update_space(self, space_id: str, space_data: SpaceUpdate, db: Collection) -> Optional[Space]:
        spaces_collection = db["spaces"]
        space_dict = space_data.dict(exclude_unset=True)
        
        # Add updated_at timestamp
        space_dict["updated_at"] = datetime.now()

        try:
            updated_space = spaces_collection.find_one_and_update(
                {"_id": ObjectId(space_id)},
                {"$set": space_dict},
                return_document=True
            )

            if updated_space:
                return Space(**updated_space)
            return None
        except Exception as e:
            raise ValueError(f"Space update failed: {str(e)}")

    async def delete_space(self, space_id: str, db: Collection) -> bool:
        spaces_collection = db["spaces"]
        space_logs_collection = db["space_logs"]
        
        # First check if the space exists
        space = await self.get_space_by_id(space_id, db)
        if not space:
            return False
            
        # Delete all associated logs first
        space_logs_collection.delete_many({"space_id": space_id})
        
        # Delete the space
        result = spaces_collection.delete_one({"_id": ObjectId(space_id)})
        
        return result.deleted_count > 0

    async def add_patient_to_space(self, space_id: str, patient_id: str, db: Collection) -> Optional[Space]:
        spaces_collection = db["spaces"]
        
        try:
            # Check if patient is already in the space
            space = await self.get_space_by_id(space_id, db)
            if not space:
                raise ValueError(f"Space with ID {space_id} does not exist")
                
            if patient_id in space.patient_ids:
                # Patient already in the space, nothing to do
                return space
                
            # Add the patient to the space
            updated_space = spaces_collection.find_one_and_update(
                {"_id": ObjectId(space_id)},
                {
                    "$addToSet": {"patient_ids": patient_id},
                    "$set": {"updated_at": datetime.now()}
                },
                return_document=True
            )
            
            if updated_space:
                return Space(**updated_space)
            return None
            
        except Exception as e:
            raise ValueError(f"Adding patient to space failed: {str(e)}")
    
    async def remove_patient_from_space(self, space_id: str, patient_id: str, db: Collection) -> Optional[Space]:
        spaces_collection = db["spaces"]
        
        try:
            # Check if space exists
            space = await self.get_space_by_id(space_id, db)
            if not space:
                raise ValueError(f"Space with ID {space_id} does not exist")
                
            # Remove the patient from the space
            updated_space = spaces_collection.find_one_and_update(
                {"_id": ObjectId(space_id)},
                {
                    "$pull": {"patient_ids": patient_id},
                    "$set": {"updated_at": datetime.now()}
                },
                return_document=True
            )
            
            if updated_space:
                return Space(**updated_space)
            return None
            
        except Exception as e:
            raise ValueError(f"Removing patient from space failed: {str(e)}")


class SpaceLogController:
    async def create_space_log(self, space_log_data: SpaceLogCreate, db: Collection) -> Optional[SpaceLog]:
        space_logs_collection = db["space_logs"]
        
        try:
            # First verify that the space exists
            spaces_collection = db["spaces"]
            space = spaces_collection.find_one({"_id": ObjectId(space_log_data.space_id)})
            
            if not space:
                raise ValueError(f"Space with ID {space_log_data.space_id} does not exist")
            
            # Create space log dictionary
            space_log_dict = space_log_data.dict()
            
            # Set default hazard_list if not provided
            if not space_log_dict.get("hazard_list"):
                space_log_dict["hazard_list"] = {
                    "high_priority": [],
                    "medium_priority": [],
                    "low_priority": []
                }
                
            space_log_dict["created_at"] = datetime.now()
            space_log_dict["updated_at"] = datetime.now()

            # Insert the space log data into the collection
            result = space_logs_collection.insert_one(space_log_dict)

            # Fetch the inserted document
            created_log = space_logs_collection.find_one(
                {"_id": result.inserted_id})

            if created_log is None:
                raise ValueError("Space log was not created successfully.")

            return SpaceLog(**created_log)

        except Exception as e:
            print(f"Error in create_space_log: {str(e)}")
            raise ValueError(f"Space log creation failed: {str(e)}")

    async def get_space_log_by_id(self, log_id: str, db: Collection) -> Optional[SpaceLog]:
        space_logs_collection = db["space_logs"]
        log_data = space_logs_collection.find_one({"_id": ObjectId(log_id)})

        if log_data:
            return SpaceLog(**log_data)
        return None

    async def get_logs_by_space_id(self, space_id: str, db: Collection) -> List[SpaceLog]:
        space_logs_collection = db["space_logs"]
        logs_data = space_logs_collection.find({"space_id": space_id}, sort=[("created_at", -1)])

        logs = []
        for log_data in logs_data:
            logs.append(SpaceLog(**log_data))
        
        return logs

    async def update_space_log(self, log_id: str, log_data: SpaceLogUpdate, db: Collection) -> Optional[SpaceLog]:
        space_logs_collection = db["space_logs"]
        log_dict = log_data.dict(exclude_unset=True)
        
        # Add updated_at timestamp
        log_dict["updated_at"] = datetime.now()

        try:
            updated_log = space_logs_collection.find_one_and_update(
                {"_id": ObjectId(log_id)},
                {"$set": log_dict},
                return_document=True
            )

            if updated_log:
                return SpaceLog(**updated_log)
            return None
        except Exception as e:
            raise ValueError(f"Space log update failed: {str(e)}")

    async def delete_space_log(self, log_id: str, db: Collection) -> bool:
        space_logs_collection = db["space_logs"]
        
        # First check if the log exists
        log = await self.get_space_log_by_id(log_id, db)
        if not log:
            return False
            
        # Delete the log
        result = space_logs_collection.delete_one({"_id": ObjectId(log_id)})
        
        return result.deleted_count > 0 