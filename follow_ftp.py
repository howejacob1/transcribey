import sys
import paramiko
from utils import parse_sftp_url

def main():
    if len(sys.argv) != 2:
        print("Usage: python follow_ftp.py sftp://username@host[:port]/path")
        sys.exit(1)

    sftp_url = parse_sftp_url(sys.argv[1])

    # Connect using paramiko and public key authentication
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(sftp_url["hostname"], port=sftp_url["port"], username=sftp_url["username"])
        sftp = client.open_sftp()
        print(f"Listing {sftp_url['path']} on {sftp_url['hostname']}:")
        for entry in sftp.listdir(sftp_url["path"]):  # Top-level directory
            print(entry)
        sftp.close()
        client.close()
    except Exception as e:
        print(f"Failed to connect or list directory: {e}")

if __name__ == "__main__":
    main()