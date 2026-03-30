"""Tests for auth_utils: password hashing and JWT token management."""

import pytest


def test_hash_password_returns_hashed_string():
    from rosettastone.server.auth_utils import hash_password

    plain = "mysecretpassword"
    result = hash_password(plain)
    assert result != plain
    assert result.startswith("$2b$")


def test_verify_password_correct():
    from rosettastone.server.auth_utils import hash_password, verify_password

    plain = "correcthorsebatterystaple"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True


def test_verify_password_wrong():
    from rosettastone.server.auth_utils import hash_password, verify_password

    hashed = hash_password("correct")
    assert verify_password("wrong", hashed) is False


def test_create_jwt_returns_string():
    from rosettastone.server.auth_utils import create_jwt

    token = create_jwt(user_id=1, role="admin", secret="testsecret")
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_jwt_roundtrip():
    from rosettastone.server.auth_utils import create_jwt, decode_jwt

    token = create_jwt(user_id=1, role="admin", secret="testsecret")
    payload = decode_jwt(token, "testsecret")
    assert payload["sub"] == "1"
    assert payload["role"] == "admin"


def test_decode_jwt_expired_token():
    import jwt

    from rosettastone.server.auth_utils import create_jwt, decode_jwt

    token = create_jwt(user_id=42, role="viewer", secret="testsecret", expires_in_seconds=-1)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_jwt(token, "testsecret")


def test_decode_jwt_invalid_token():
    import jwt

    from rosettastone.server.auth_utils import decode_jwt

    with pytest.raises(jwt.InvalidTokenError):
        decode_jwt("not.a.token", "testsecret")


def test_decode_jwt_wrong_secret():
    import jwt

    from rosettastone.server.auth_utils import create_jwt, decode_jwt

    token = create_jwt(user_id=7, role="editor", secret="secret1")
    with pytest.raises(jwt.InvalidSignatureError):
        decode_jwt(token, "secret2")
