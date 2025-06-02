from mongo_utils import get_mongo_collection
from bson import ObjectId # To check for ObjectId type if necessary
import vcon # Import the vcon library

def print_vcon_details():
    vcons_collection = get_mongo_collection()
    
    print("Fetching vCon information using vcon library...")
    
    for vcon_doc_from_db in vcons_collection.find():
        # Prepare the dictionary for vcon.from_dict()
        # The Vcon object uses 'uuid', MongoDB uses '_id' by default.
        # vcon_utils.py should have stored a 'uuid' from vcon_obj.to_dict().
        dict_for_load = vcon_doc_from_db.copy()
        doc_id_for_print = str(dict_for_load.get("_id", "UNKNOWN_ID"))
        dict_for_load.pop('_id', None) # Remove MongoDB _id, rely on uuid within the doc

        try:
            v_obj = vcon.Vcon(dict_for_load)
            # If from_dict worked, v_obj.uuid should be populated.
            # If 'uuid' was not in dict_for_load, from_dict might raise error or uuid might be None.
            if not v_obj.uuid:
                # Fallback if uuid wasn't in the dict or not set by from_dict correctly
                # This case implies an issue with how vCons are stored/structured by vcon_utils or main.py
                v_obj.uuid = doc_id_for_print 
                print(f"Warning: vCon loaded from DB (original _id: {doc_id_for_print}) did not have a UUID in its dictionary representation or from_dict did not set it. Using original _id as UUID for printing.")

        except Exception as e:
            print(f"--- vCon DB _id (raw): {doc_id_for_print} ---")
            print(f"  Error loading DB doc into Vcon object: {e}")
            print(f"  Raw document from DB: {vcon_doc_from_db}")
            print("-" * (40 + len(doc_id_for_print)))
            print()
            continue

        vcon_id_to_display = v_obj.uuid
        transcriptions = []
        languages = []
        
        # Access analysis data through the vcon object's properties
        # Assuming v_obj.analysis is a list of dicts as per vCon structure
        if hasattr(v_obj, 'analysis') and isinstance(v_obj.analysis, list):
            for analysis_entry in v_obj.analysis:
                analysis_type = analysis_entry.get("type")
                analysis_body = analysis_entry.get("body")
                
                if analysis_type == "transcription":
                    if isinstance(analysis_body, str):
                        transcriptions.append(analysis_body)
                    elif isinstance(analysis_body, list): # Should not happen if body is just text
                        transcriptions.extend([str(item) for item in analysis_body]) 
                elif analysis_type == "language_identification":
                    if isinstance(analysis_body, list):
                        languages.extend(analysis_body)
                    elif isinstance(analysis_body, str): # e.g. "en"
                        languages.append(analysis_body)
        else:
            # This case means the Vcon object (as understood by the library) has no .analysis list
            print(f"Note: Vcon object for UUID {vcon_id_to_display} (original _id: {doc_id_for_print}) has no 'analysis' attribute or it's not a list after from_dict.")

        print(f"--- vCon UUID: {vcon_id_to_display} (DB _id: {doc_id_for_print}) ---")
        if languages:
            flat_languages = []
            for lang_item in languages:
                if isinstance(lang_item, list): # e.g. ["en", "es"]
                    flat_languages.extend(lang_item)
                else: # e.g. "en"
                    flat_languages.append(lang_item)
            unique_languages = sorted(list(set(flat_languages)))
            print(f"  Identified Languages: {unique_languages}")
        else:
            print("  Identified Languages: Not found")
            
        if transcriptions:
            print("  Transcriptions:")
            for i, trans_text in enumerate(transcriptions):
                print(f"    [{i+1}]: {trans_text}")
        else:
            print("  Transcriptions: Not found")
        print("-" * (40 + len(str(vcon_id_to_display))))
        print()

if __name__ == "__main__":
    print_vcon_details() 