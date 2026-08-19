"""One visual contract shared by every generated COLLBRAI content video."""

from __future__ import annotations

from dataclasses import dataclass


BRAND_NAME = "COLT AI - COLLBRAI"
THEME_ID = "colt-collbrai-navy-v1"


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """Convert a strict ``#RRGGBB`` token to OpenCV BGR order."""

    if (
        not isinstance(value, str)
        or len(value) != 7
        or not value.startswith("#")
    ):
        raise ValueError("color must use #RRGGBB")
    try:
        red = int(value[1:3], 16)
        green = int(value[3:5], 16)
        blue = int(value[5:7], 16)
    except ValueError as exc:
        raise ValueError("color must use #RRGGBB") from exc
    return blue, green, red


@dataclass(frozen=True)
class VisualTheme:
    theme_id: str = THEME_ID
    brand_name: str = BRAND_NAME
    background: str = "#06152D"
    panel: str = "#0D2B52"
    text: str = "#F4F7FF"
    muted_text: str = "#A8B9D4"
    safe: str = "#46C7FF"
    warning: str = "#789DFF"
    violation: str = "#FF637D"

    def as_dict(self) -> dict[str, str]:
        return {
            "theme_id": self.theme_id,
            "brand_name": self.brand_name,
            "background": self.background,
            "panel": self.panel,
            "text": self.text,
            "muted_text": self.muted_text,
            "safe": self.safe,
            "warning": self.warning,
            "violation": self.violation,
        }

    def bgr(self, token: str) -> tuple[int, int, int]:
        if token not in {
            "background",
            "panel",
            "text",
            "muted_text",
            "safe",
            "warning",
            "violation",
        }:
            raise KeyError(token)
        return hex_to_bgr(self.as_dict()[token])


THEME = VisualTheme()
