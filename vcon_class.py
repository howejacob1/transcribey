import datetime
import mimetypes
import settings
import uuid
import torch
import numpy as np
from typing import List
from utils import suppress_output
from vcon import Vcon as VconBase
from vcon.dialog import Dialog
from vcon.party import Party
import audio
import gpu

class Vcon(VconBase):
    def __init__(self, vcon_dict=None, property_handling=None):
        super().__init__(vcon_dict, property_handling)
        self.vcon_dict["vcon"] = "0.0.2"
        
        # Ensure dialog is always a list
        if "dialog" not in self.vcon_dict:
            self.vcon_dict["dialog"] = []
        elif not isinstance(self.vcon_dict["dialog"], list):
            self.vcon_dict["dialog"] = []
        
        # Ensure analysis is always a list
        if "analysis" not in self.vcon_dict:
            self.vcon_dict["analysis"] = []
        elif not isinstance(self.vcon_dict["analysis"], list):
            self.vcon_dict["analysis"] = []

    @classmethod
    def build_new(cls):
        """Create a new Vcon instance"""
        # Use the parent's build_new method which properly initializes all fields including UUID
        vcon_base = VconBase.build_new()
        # Create our custom class instance from the properly initialized vcon_dict
        vcon_base.vcon_dict["dialog"] = []
        vcon_base.vcon_dict["analysis"] = []

        return cls(vcon_dict=vcon_base.vcon_dict)

    @classmethod
    def from_dict(cls, vcon_dict):
        """Create a Vcon instance from a dictionary (standard vcon format)"""
        # Validate and fix the input dictionary before creating the instance
        if vcon_dict is None:
            vcon_dict = {}
        
        # Make a copy to avoid modifying the original
        safe_dict = dict(vcon_dict)
        
        # Ensure dialog field is always a list
        if "dialog" not in safe_dict:
            safe_dict["dialog"] = []
        elif not isinstance(safe_dict["dialog"], list):
            safe_dict["dialog"] = []
        
        # Ensure analysis field is always a list  
        if "analysis" not in safe_dict:
            safe_dict["analysis"] = []
        elif not isinstance(safe_dict["analysis"], list):
            safe_dict["analysis"] = []
            
        with suppress_output(should_suppress=True):
            return cls(vcon_dict=safe_dict)

    @classmethod
    def create_from_url(cls, url):
        """Create a new Vcon instance from a URL (class method version)"""
        with suppress_output(should_suppress=True):
            vcon = cls.build_new()
            vcon._setup_from_url(url)
            return vcon

    def to_dict(self):
        """Convert the Vcon to a dictionary"""
        return super().to_dict()

    @property
    def uuid(self):
        """Get the UUID of the vcon"""
        return super().uuid

    @property
    def size(self):
        """Get the size of the vcon"""
        try:
            if not self.vcon_dict.get("dialog") or len(self.vcon_dict["dialog"]) == 0:
                return None
            return self.vcon_dict["dialog"][0]["size_bytes"]
        except (KeyError, IndexError, TypeError):
            return None

    @size.setter
    def size(self, value: int):
        """Set the size of the vcon"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{"type": "recording"}]
        self.vcon_dict["dialog"][0]["size_bytes"] = value

    @property
    def duration(self):
        """Get the duration of the vcon"""
        try:
            if not self.vcon_dict.get("dialog") or len(self.vcon_dict["dialog"]) == 0:
                return 0
            return self.vcon_dict["dialog"][0]["duration"]
        except (KeyError, IndexError, TypeError):
            return 0

    @duration.setter
    def duration(self, value: float):
        """Set the duration of the vcon"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{"type": "recording"}]
        self.vcon_dict["dialog"][0]["duration"] = value


    @property
    def filename(self):
        """Get the filename from the first dialog"""
        try:
            if not self.vcon_dict.get("dialog", None) or len(self.vcon_dict["dialog"]) == 0:
                return None
            return self.vcon_dict["dialog"][0]["filename"]
        except (KeyError, IndexError, TypeError):
            return None

    @filename.setter
    def filename(self, value):
        """Set the filename in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{"type": "recording", "filename": value, "encoding": "none"}]
        self.vcon_dict["dialog"][0]["filename"] = value







    @property
    def basename(self):
        """Get the basename from the first dialog"""
        try:
            if not self.vcon_dict.get("dialog", None) or len(self.vcon_dict["dialog"]) == 0:
                return None
            return self.vcon_dict["dialog"][0]["basename"]
        except (KeyError, IndexError, TypeError):
            return None

    @basename.setter
    def basename(self, value):
        """Set the basename in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{"type": "recording", "basename": value, "encoding": "none"}]
        self.vcon_dict["dialog"][0]["basename"] = value









    @property
    def audio(self):
        """Get the audio data from the first dialog"""
        try:
            if not self.vcon_dict.get("dialog") or len(self.vcon_dict["dialog"]) == 0:
                return None
            return self.vcon_dict["dialog"][0]["body"]
        except (KeyError, IndexError, TypeError):
            return None

    @audio.setter
    def audio(self, value):
        """Set the audio data in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{"type": "recording"}]
        self.vcon_dict["dialog"][0]["body"] = value

    def find_transcript_analysis(self):
        """get the transcript analysis"""
        try:
            for analysis in self.vcon_dict["analysis"]:
                if analysis["type"] == "transcript":
                    return analysis
            return None
        except (KeyError, IndexError, TypeError):
            return None

    def find_language_identification_analysis(self):
        """Get the transcript text from the first dialog"""
        try:
            for analysis in self.vcon_dict["analysis"]:
                if analysis["type"] == "language_identification":
                    return analysis
            return None
        except (KeyError, IndexError, TypeError):
            return None

    def transcript(self):
        """Get the transcript text from the first dialog"""
        analysis = self.find_transcript_analysis()
        if analysis:
            return analysis["body"]["text"]
        return None

    def set_transcript(self, value, model):
        self.vcon_dict["analysis"].append({"type":"transcript",
                                            "body": {"text":value},
                                           "vendor":"bantaim"})

    @property
    def languages(self):
        """Get the languages from the first dialog transcript"""
        analysis = self.find_language_identification_analysis()
        if analysis:
            languages_data = analysis["body"]["languages"]
            # Handle both dict and list formats
            if isinstance(languages_data, dict):
                return list(languages_data.keys())
            elif isinstance(languages_data, list):
                return languages_data
            else:
                return None
        return None

    @languages.setter
    def languages(self, languages: List[str]):
        """Set the languages in the first dialog transcript"""
        self.vcon_dict["analysis"].append({"type":"language_identification", "body":{"languages":languages},
                                           "vendor":"bantaim"})

    @property
    def done(self):
        """Get the done status of the vcon"""
        return self.vcon_dict.get("done", False)

    @done.setter
    def done(self, value):
        """Set the done status of the vcon"""
        self.vcon_dict["done"] = value

    @property
    def sample_rate(self):
        """Get the sample rate of the vcon"""
        try:
            if not self.vcon_dict.get("dialog") or len(self.vcon_dict["dialog"]) == 0:
                return None
            return self.vcon_dict["dialog"][0]["sample_rate"]
        except (KeyError, IndexError, TypeError):
            return None

    @sample_rate.setter
    def sample_rate(self, value):
        """Set the sample rate of the vcon"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{"type": "recording"}]
        self.vcon_dict["dialog"][0]["sample_rate"] = value

    @property
    def corrupt(self):
        """Get the corrupt status of the vcon"""
        return self.vcon_dict.get("corrupt", False)

    @corrupt.setter
    def corrupt(self, value):
        """Set the corrupt status of the vcon"""
        self.vcon_dict["corrupt"] = value

    @property
    def processed_by(self):
        """Get the processed_by field of the vcon"""
        return self.vcon_dict.get("processed_by")

    @processed_by.setter
    def processed_by(self, value):
        """Set the processed_by field of the vcon"""
        self.vcon_dict["processed_by"] = value


    def mark_as_done(self):
        """Mark the vcon as done"""
        self.done = True

    def mark_as_invalid(self):
        """Mark the vcon as corrupt and done"""
        self.corrupt = True
        self.done = True

    def set_languages(self, languages):
        """Set the languages for this vcon"""
        self.languages = languages

    def get_languages(self):
        """Get the languages for this vcon"""
        return self.languages

    def _setup_from_url(self, url):
        """Setup a vcon from a URL (internal method)"""
        # Add a party
        party = Party(name="Unknown", role="participant")
        self.add_party(party)
        
        # Create a dialog - set it directly in vcon_dict to ensure it works
        now = datetime.datetime.now(datetime.timezone.utc)
        mimetype, _ = mimetypes.guess_type(url)
        
        # Initialize dialog structure if it doesn't exist
        if "dialog" not in self.vcon_dict:
            self.vcon_dict["dialog"] = []
        
        # Create the dialog directly in the dict
        dialog_dict = {
            "type": "recording",
            "start": now.isoformat(),
            "parties": [0],
            "originator": 0,
            "mediatype": mimetype,
            "filename": url,
            "body": None,
            "encoding": "none",
        }
        
        # Replace or add the first dialog entry
        if len(self.vcon_dict["dialog"]) == 0:
            self.vcon_dict["dialog"].append(dialog_dict)
        else:
            self.vcon_dict["dialog"][0] = dialog_dict
        
        return self

    def setup_from_url(self, url):
        """Setup a vcon from a URL (public method)"""
        return self._setup_from_url(url)

    def __repr__(self):
        return self._summary_str()

    def __str__(self):
        return self._summary_str()

    def _summary_str(self):
        # Transcript
        transcript_str = ""
        transcript_str = self.transcript()
        if transcript_str is not None:
            if len(transcript_str) > 50:
                transcript_str = transcript_str[:50] + "..."

        # Duration, handling None
        duration_str = f"{self.duration:.2f}s" if self.duration is not None else "N/A"

        # Size, handling None
        size_str = audio.format_bytes(self.size) if self.size is not None else "N/A"

        # Languages, handling None or empty
        languages_str = ', '.join(self.languages) if self.languages else "N/A"

        # UUID, showing first 8 chars
        uuid_str = str(self.uuid) if self.uuid else "N/A"

        return (f"Vcon({uuid_str}, {duration_str}, {size_str}, "
                f"lang={languages_str}, transcript='{transcript_str}')")

    # Dictionary interface methods for MongoDB compatibility
    def __getitem__(self, key):
        """Make the Vcon behave like a dict for MongoDB serialization"""
        return self.vcon_dict[key]
    
    def __setitem__(self, key, value):
        """Allow dict-like assignment"""
        self.vcon_dict[key] = value
    
    def __delitem__(self, key):
        """Allow dict-like deletion"""
        del self.vcon_dict[key]
    
    def __iter__(self):
        """Allow iteration over keys like a dict"""
        return iter(self.vcon_dict)
    
    def __len__(self):
        """Return the length like a dict"""
        return len(self.vcon_dict)
    
    def keys(self):
        """Return keys like a dict"""
        return self.vcon_dict.keys()
    
    def values(self):
        """Return values like a dict"""
        return self.vcon_dict.values()
    
    def items(self):
        """Return items like a dict"""
        return self.vcon_dict.items()
    
    def get(self, key, default=None):
        """Get method like a dict"""
        return self.vcon_dict.get(key, default)
    
    def __contains__(self, key):
        """Support 'in' operator like a dict"""
        return key in self.vcon_dict

    def add_party(self, party):
        # Use the parent class method which properly manages the parties list
        super().add_party(party)

    def add_dialog(self, dialog):
        # Use the parent class method which properly manages the dialog list
        super().add_dialog(dialog)



