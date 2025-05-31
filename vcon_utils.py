import datetime
import logging
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog

def create_vcon_for_wav(url):
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
    logging.getLogger().setLevel(logging.INFO)
    vcon_dict = vcon.to_dict()
    vcon_dict["filename"] = url
    return vcon_dict
