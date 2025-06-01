import datetime
import logging
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog
from utils import get_file_size

def create_vcon_for_wav(url, sftp_client):
    """ 
    Create a vCon for the given wav file URL and return it as a dict.
    """
    logging.getLogger().setLevel(logging.WARN)
    vcon = Vcon.build_new()
    party = Party(name="Unknown", role="participant")
    vcon.add_party(party)
    now = datetime.datetime.now(datetime.timezone.utc)
    dialog = Dialog(
        type="audio",
        start=now.isoformat(),
        parties=[0],
        originator=0,
        mimetype="audio/wav",
        filename=url,
        body=None,
        encoding=None
    )
    vcon.add_dialog(dialog)
    vcon.add_attachment(type="audio", body=url, encoding="none")
    size = get_file_size(url, sftp_client)
    logging.getLogger().setLevel(logging.INFO)
    vcon_dict = vcon.to_dict()
    vcon_dict["filename"] = url
    vcon_dict["size"] = size
    return vcon_dict
