"""File-based per-user password authentication for the Chemprop web app.

Credentials live in a JSON file (no database), mapping each username to a
Werkzeug password hash. The authenticated identity is kept in the Flask session
cookie, which is signed with SECRET_KEY and therefore cannot be forged
client-side (unlike the legacy ``currentUser`` cookie).
"""

import json
import os
import secrets
from typing import Dict

from werkzeug.security import check_password_hash, generate_password_hash

from chemprop.web.app import app


def _credentials_path() -> str:
    return app.config['CREDENTIALS_PATH']


def load_credentials() -> Dict[str, str]:
    """Returns the username -> password-hash map, or an empty map if none exists."""
    path = _credentials_path()

    if not os.path.isfile(path):
        return {}

    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_credentials(credentials: Dict[str, str]) -> None:
    """Writes the credentials map atomically with owner-only permissions."""
    path = _credentials_path()
    tmp_path = f'{path}.tmp'

    with open(tmp_path, 'w') as f:
        json.dump(credentials, f, indent=2)
    os.replace(tmp_path, path)
    os.chmod(path, 0o600)


def set_password(username: str, password: str) -> None:
    """Creates or updates the stored password hash for a user."""
    credentials = load_credentials()
    credentials[username] = generate_password_hash(password)
    save_credentials(credentials)


def verify_password(username: str, password: str) -> bool:
    """Returns True if the username exists and the password matches its hash."""
    hashed = load_credentials().get(username)
    return bool(hashed) and check_password_hash(hashed, password)


def user_exists(username: str) -> bool:
    """Returns True if a credential entry exists for the given username."""
    return username in load_credentials()


def load_or_create_secret_key(path: str) -> str:
    """Returns the session-signing key.

    Prefers the ``CHEMPROP_SECRET_KEY`` environment variable; otherwise reads a
    persisted key from ``path``, generating and storing one on first use so that
    sessions survive app restarts.
    """
    env_key = os.environ.get('CHEMPROP_SECRET_KEY')
    if env_key:
        return env_key

    if os.path.isfile(path):
        with open(path) as f:
            return f.read().strip()

    key = secrets.token_hex(32)
    with open(path, 'w') as f:
        f.write(key)
    os.chmod(path, 0o600)
    return key
