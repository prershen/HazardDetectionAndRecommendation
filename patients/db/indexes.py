from pymongo import ASCENDING, IndexModel
from pymongo.collection import Collection


def create_patient_indexes(db):
    """
    Create indexes for the patients collection.
    """
    patients_collection = db["patients"]
    
    # Create indexes
    user_id_index = IndexModel([("user_id", ASCENDING)], 
                             name="user_id_index")
    
    # Create compound index for querying patients by user_id and patient_name
    user_patient_index = IndexModel([("user_id", ASCENDING), 
                                   ("patient_name", ASCENDING)], 
                                  name="user_patient_index")
    
    # Create the indexes
    patients_collection.create_indexes([user_id_index, user_patient_index])
    
    print("Patient indexes created successfully") 