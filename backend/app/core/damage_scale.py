"""The four xBD damage classes and the single ordinal colour ramp used everywhere for them.

The ramp is one hue (plum), light → dark with severity, validated as a colour-blind-safe ordinal ramp
on a light surface. Colour is never the only cue: figures and UI always pair it with the class name.
"""

DAMAGE_CLASSES: tuple[str, ...] = ("no-damage", "minor-damage", "major-damage", "destroyed")

DAMAGE_COLORS: dict[str, str] = {
    "no-damage": "#c9a8d8",
    "minor-damage": "#9a64b5",
    "major-damage": "#6b3590",
    "destroyed": "#3d1559",
}
