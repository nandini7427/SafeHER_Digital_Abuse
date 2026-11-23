from argon2 import PasswordHasher
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Return an Argon2 password hash."""
    if not password:
        raise ValueError("Password cannot be empty.")

    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its stored Argon2 hash."""
    if not plain_password or not hashed_password:
        return False

    try:
        return password_hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
