"""Vector icon rendering and key-name formatting utilities."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPixmap


def create_gear_icon(size: int = 64) -> QIcon:
    """Render a vector gear icon as a QIcon.

    Args:
        size: Pixel dimensions of the square icon.

    Returns:
        A QIcon containing the rendered gear.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Colors
    color = QColor("#ffffff")
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))

    center = QPointF(size / 2, size / 2)
    outer_radius = size * 0.45
    inner_radius = size * 0.25
    teeth = 8

    path = QPainterPath()
    path.addEllipse(center, inner_radius, inner_radius)
    path.addEllipse(center, outer_radius, outer_radius)  # Ring

    # Draw teeth
    for _i in range(teeth):
        pass

    # Let's draw a proper gear shape
    gear_path = QPainterPath()
    gear_path.setFillRule(Qt.WindingFill)

    r_out = size * 0.45
    r_in = size * 0.35
    r_hole = size * 0.15

    for i in range(teeth * 2):
        angle = 2 * math.pi * i / (teeth * 2)
        r = r_out if i % 2 == 0 else r_in
        x = center.x() + r * math.cos(angle)
        y = center.y() + r * math.sin(angle)
        if i == 0:
            gear_path.moveTo(x, y)
        else:
            gear_path.lineTo(x, y)
    gear_path.closeSubpath()

    # Subtract hole
    hole_path = QPainterPath()
    hole_path.addEllipse(center, r_hole, r_hole)
    final_path = gear_path.subtracted(hole_path)

    painter.drawPath(final_path)
    painter.end()
    return QIcon(pixmap)


def format_key_name(key_str: str) -> str:
    """Convert a raw pynput key string to a human-readable name.

    Args:
        key_str: The internal key identifier (e.g. "keyboard.Key.shift_r").

    Returns:
        A user-friendly display name (e.g. "Right Shift").
    """
    if not key_str:
        return ""

    key_map = {
        "keyboard.Key.shift_r": "Right Shift",
        "keyboard.Key.shift": "Left Shift",
        "keyboard.Key.ctrl_l": "Left Ctrl",
        "keyboard.Key.ctrl_r": "Right Ctrl",
        "keyboard.Key.alt_l": "Left Alt",
        "keyboard.Key.alt_r": "Right Alt",
        "keyboard.Key.esc": "Escape",
        "keyboard.Key.space": "Space",
        "keyboard.Key.enter": "Enter",
        "keyboard.Key.tab": "Tab",
        "keyboard.Key.caps_lock": "Caps Lock",
        "keyboard.Key.cmd": "Windows/Cmd",
        "keyboard.Key.f1": "F1",
        "keyboard.Key.f2": "F2",
        "keyboard.Key.f3": "F3",
        "keyboard.Key.f4": "F4",
        "keyboard.Key.f5": "F5",
        "keyboard.Key.f6": "F6",
        "keyboard.Key.f7": "F7",
        "keyboard.Key.f8": "F8",
        "keyboard.Key.f9": "F9",
        "keyboard.Key.f10": "F10",
        "keyboard.Key.f11": "F11",
        "keyboard.Key.f12": "F12",
    }
    # Handle single characters (e.g., 'a', '1')
    if key_str.startswith("'") and key_str.endswith("'") and len(key_str) == 3:
        return key_str[1].upper()

    return key_map.get(
        key_str, key_str.replace("keyboard.Key.", "").replace("_", " ").title()
    )
