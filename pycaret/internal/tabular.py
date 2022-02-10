from typing import Optional


def _get_setup_signature() -> Optional[str]:
    return globals().get("_setup_signature", None)
