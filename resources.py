"""
Crisis & support resources, region-aware.

IMPORTANT: Helpline numbers change. Do NOT treat this file as a source of truth
forever. Wire it to a maintained source on a schedule. ThroughLine's
findahelpline.com (https://findahelpline.com) offers verified, internationally
maintained listings and an API — strongly recommended as the canonical source
so a user is never shown a dead number in a crisis.

Numbers below verified June 2026.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Resource:
    name: str
    call: Optional[str] = None
    text: Optional[str] = None
    chat_url: Optional[str] = None
    note: str = ""
    hours: str = "24/7"


# Keyed by ISO 3166-1 alpha-2 country code. Fall back to GLOBAL.
RESOURCES: dict[str, list[Resource]] = {
    "IN": [
        Resource(
            name="Tele-MANAS (Govt. of India)",
            call="14416",  # or 1800-891-4416
            note="Free national mental-health helpline, 20+ languages. Can arrange "
                 "video consult / e-prescription with consent.",
        ),
    ],
    "US": [
        Resource(
            name="988 Suicide & Crisis Lifeline",
            call="988",
            text="988",
            chat_url="https://988lifeline.org/chat/",
            note="Free, confidential. Veterans press 1; Spanish press 2. Does not "
                 "auto-dispatch police; escalates only when necessary.",
        ),
    ],
    "GLOBAL": [
        Resource(
            name="Find A Helpline",
            chat_url="https://findahelpline.com",
            note="Verified crisis lines for 130+ countries. Use to look up local support.",
        ),
    ],
}


def for_country(country_code: Optional[str]) -> list[Resource]:
    cc = (country_code or "").upper()
    local = RESOURCES.get(cc, [])
    # Always append the global lookup so the user is never left without an option.
    return local + RESOURCES["GLOBAL"]
