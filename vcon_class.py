from vcon import Vcon as VconBase
from vcon.dialog import Dialog
from vcon.party import Party
import uuid
import datetime
import mimetypes
import audio

class Vcon(VconBase):
    def __init__(self, vcon_dict=None, property_handling=None):
        super().__init__(vcon_dict, property_handling)

    @classmethod
    def build_new(cls):
        """Create a new Vcon instance"""
        return cls()

    @classmethod
    def from_dict(cls, vcon_dict):
        """Create a Vcon instance from a dictionary (standard vcon format)"""
        return cls(vcon_dict=vcon_dict)

    @classmethod
    def create_from_url(cls, url):
        """Create a new Vcon instance from a URL (class method version)"""
        vcon = cls.build_new()
        vcon._setup_from_url(url)
        return vcon

    def to_dict(self):
        """Convert the Vcon to a dictionary"""
        return self.vcon_dict

    @property
    def uuid(self):
        """Get the UUID of the vcon"""
        return self.vcon_dict.get("uuid")

    @property
    def size(self):
        """Get the size of the vcon"""
        return self.vcon_dict.get("size")

    @size.setter
    def size(self, value):
        """Set the size of the vcon"""
        self.vcon_dict["size"] = value

    @property
    def filename(self):
        """Get the filename from the first dialog"""
        try:
            return self.vcon_dict["dialog"][0]["filename"]
        except (KeyError, IndexError, TypeError):
            return None

    @filename.setter
    def filename(self, value):
        """Set the filename in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{}]
        self.vcon_dict["dialog"][0]["filename"] = value

    @property
    def audio(self):
        """Get the audio data from the first dialog"""
        try:
            return self.vcon_dict["dialog"][0]["body"]
        except (KeyError, IndexError, TypeError):
            return None

    @audio.setter
    def audio(self, value):
        """Set the audio data in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{}]
        self.vcon_dict["dialog"][0]["body"] = value

    @property
    def transcript_text(self):
        """Get the transcript text from the first dialog"""
        try:
            return self.vcon_dict["dialog"][0]["transcript"]["text"]
        except (KeyError, IndexError, TypeError):
            return None

    @transcript_text.setter
    def transcript_text(self, value):
        """Set the transcript text in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{}]
        if "transcript" not in self.vcon_dict["dialog"][0]:
            self.vcon_dict["dialog"][0]["transcript"] = {}
        self.vcon_dict["dialog"][0]["transcript"]["text"] = value

    @property
    def languages(self):
        """Get the languages from the first dialog transcript"""
        try:
            return self.vcon_dict["dialog"][0]["transcript"]["languages"]
        except (KeyError, IndexError, TypeError):
            return None

    @languages.setter
    def languages(self, value):
        """Set the languages in the first dialog transcript"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{}]
        if "transcript" not in self.vcon_dict["dialog"][0]:
            self.vcon_dict["dialog"][0]["transcript"] = {}
        self.vcon_dict["dialog"][0]["transcript"]["languages"] = value

    @property
    def transcript_dict(self):
        """Get the entire transcript dictionary from the first dialog"""
        try:
            return self.vcon_dict["dialog"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            return None

    @transcript_dict.setter
    def transcript_dict(self, value):
        """Set the entire transcript dictionary in the first dialog"""
        if "dialog" not in self.vcon_dict or not self.vcon_dict["dialog"]:
            self.vcon_dict["dialog"] = [{}]
        self.vcon_dict["dialog"][0]["transcript"] = value

    @property
    def transcript(self):
        """Alias for transcript_text for compatibility"""
        return self.transcript_text

    @transcript.setter
    def transcript(self, value):
        """Alias for transcript_text setter for compatibility"""
        self.transcript_text = value

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
        return self.vcon_dict.get("sample_rate")

    @sample_rate.setter
    def sample_rate(self, value):
        """Set the sample rate of the vcon"""
        self.vcon_dict["sample_rate"] = value

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

    def is_mono(self):
        """Check if the audio is mono"""
        if self.audio is None:
            return False
        return audio.is_mono(self.audio)

    def mark_as_done(self):
        """Mark the vcon as done"""
        self.done = True

    def mark_as_invalid(self):
        """Mark the vcon as corrupt and done"""
        self.corrupt = True
        self.done = True

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
        
        # Add the dialog directly to the dict
        dialog_dict = {
            "type": "audio",
            "start": now.isoformat(),
            "parties": [0],
            "originator": 0,
            "mimetype": mimetype,
            "filename": url,
            "body": None,
            "encoding": None,
            "transcript": {}
        }
        self.vcon_dict["dialog"].append(dialog_dict)
        
        return self

    def setup_from_url(self, url):
        """Setup a vcon from a URL (public method)"""
        return self._setup_from_url(url)

    def __repr__(self):
        return f"Vcon(uuid={self.uuid}, filename={self.filename}, done={self.done})"

    def __str__(self):
        return self.__repr__()

    def add_party(self, party):
        self.parties.append(party)

    def add_dialog(self, dialog):
        self.dialog.append(dialog)
