import os
import toml
from utils import what_directory_are_we_in

secrets_path = os.path.join(what_directory_are_we_in(), '.secrets.toml')
secrets = None
with open(secrets_path, 'r') as f:
    secrets = toml.load(f)
