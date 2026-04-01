"""Authentication utility functions: password hashing and JWT token management."""

from datetime import UTC, datetime, timedelta


def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt via passlib."""
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return str(pwd_context.hash(plain))


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return bool(pwd_context.verify(plain, hashed))


def create_jwt(user_id: int, role: str, secret: str, expires_in_seconds: int = 3600) -> str:
    """Create a signed JWT token with user_id, role, and expiry."""
    import jwt

    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
    }
    return str(jwt.encode(payload, secret, algorithm="HS256"))


def decode_jwt(token: str, secret: str) -> dict[str, object]:
    """Decode and validate a JWT token.

    Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError on failure.
    """
    import jwt

    return dict(jwt.decode(token, secret, algorithms=["HS256"]))
