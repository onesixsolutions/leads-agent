from pydantic import SecretStr


def mask_secret(secret: SecretStr | None, visible: int = 4) -> str:
    """Mask a secret string, handling SecretStr or None."""
    if secret is None:
        return "[not set]"
    # Handle pydantic SecretStr
    val = (
        secret.get_secret_value()
        if hasattr(secret, "get_secret_value")
        else str(secret)
    )
    if len(val) <= visible:
        return "***"
    return val[:visible] + "*" * (len(val) - visible)
