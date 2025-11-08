import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from auth import hash_password  # noqa: E402

if __name__ == "__main__":
    password = getpass.getpass("Enter password to hash: ")
    if not password:
        raise SystemExit("Password cannot be empty")
    print(hash_password(password))
