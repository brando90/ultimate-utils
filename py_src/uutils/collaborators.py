"""Brando's research collaborators: bidirectional name ↔ email ↔ github mapping.

Purpose
-------
Give agents (Claude Code, Codex, etc.) a canonical roster of who Brando
works with, so they can resolve a name/email/github handle to the person
and pick the right contact when emailing, @-mentioning, or assigning PRs.

Source
------
Derived 2026-04-21 from `git log` across ~48 repos under ``~/`` where Brando
is the majority contributor (≥50% of commits). Forks of upstream projects
(lm-evaluation-harness, harbor, mathlib, etc.) were excluded — their
contributors are not Brando's collaborators.

Schema
------
Each entry in ``COLLABORATORS`` is a dict with:
    - ``name``:    canonical display name
    - ``emails``:  list of known emails (personal, institutional, cluster hosts)
    - ``github``:  github handle (string) or None
    - ``aliases``: alternate display names seen in git log (for dedup)
    - ``affil``:   short affiliation hint (optional, best-effort)

Derived lookup tables:
    - ``by_email``    email        → record
    - ``by_name``     canonical name → record
    - ``by_github``   github handle → record
    - ``by_alias``    alias name    → record (merged with by_name for convenience)

Usage
-----
    from uutils.collaborators import by_email, by_github, resolve

    resolve("patricky168@gmail.com")   # → {"name": "Patrick Yu", ...}
    resolve("SudharsanSundar")         # github handle lookup
    resolve("Ethan Hersch")            # name lookup

Caveats
-------
- Not every author in git log is here; this is a curated roster, not exhaustive.
- Brando himself is included as ``_self`` so agents can recognise all his aliases
  (multiple emails + hostnames) as the same identity.
- Affiliations are best-effort from context; update when you learn better.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Self (all of Brando's known aliases — so agents don't mistake him for a
# collaborator when they see miranda9@ibm.com or brando9@<host>.stanford.edu)
# ---------------------------------------------------------------------------

_SELF = {
    "name": "Brando Miranda",
    "emails": [
        "brandojazz@gmail.com",
        "brando.science@gmail.com",
        "brando9@stanford.edu",
        "brando9@cs.stanford.edu",
        "brando90@mit.edu",
        "miranda9@illinois.edu",
        "miranda9@ibm.com",
        "miranebr@amazon.com",
    ],
    "github": "brando90",
    "aliases": [
        "brando90",
        "BrandoPareto",
        "Brando Miranda Garay",
        "miranda9",
        "Brando",
    ],
    "affil": "Stanford (Koyejo Lab)",
    "role": "self",
}


# ---------------------------------------------------------------------------
# Research collaborators (curated)
# ---------------------------------------------------------------------------

COLLABORATORS: list[dict[str, Any]] = [
    _SELF,
    {
        "name": "Rylan Schaeffer",
        "emails": ["rylanschaeffer@gmail.com"],
        "github": "RylanSchaeffer",
        "aliases": ["Rylan"],
        "affil": "Stanford (former); FieteLab",
    },
    {
        "name": "Patrick Yu",
        "emails": [
            "patricky168@gmail.com",
            "pzy2@vision-submit.cs.illinois.edu",
            "pzy2@vision-21.cs.illinois.edu",
            "pzy2@vision-22.cs.illinois.edu",
            "pzy2@vision-23.cs.illinois.edu",
            "pzy2@vision-02.cs.illinois.edu",
            "pzy2@vision-01.cs.illinois.edu",
            "pzy2@vision-03.cs.illinois.edu",
            "pzy2@vision-16.cs.illinois.edu",
        ],
        "github": "patricks-lab",
        "aliases": ["patrickyu", "Patrick Yu"],
        "affil": "UIUC",
    },
    {
        "name": "Sudharsan Sundar",
        "emails": ["sudharsan.j.sundar@gmail.com"],
        "github": "SudharsanSundar",
        "aliases": ["SudharsanSundar"],
        "affil": "Stanford",
    },
    {
        "name": "Kai Fronsdal",
        "emails": [
            "kaifronsdal@gmail.com",
            "kaif@stanford.edu",
            "kaif@turing2.stanford.edu",
        ],
        "github": "kaifronsdal",
        "aliases": ["kaifronsdal"],
        "affil": "Stanford",
    },
    {
        "name": "Ethan Hersch",
        "emails": ["herschethan@gmail.com"],
        "github": "ehersch",
        "aliases": ["Ethan Hersch"],
        "affil": "Stanford",
    },
    {
        "name": "Srivatsava Daruru",
        "emails": [
            "vatsava@gmail.com",
            "srivatsavad@skampere1.stanford.edu",
            "srivatsavad@skampere2.stanford.edu",
        ],
        "github": None,
        "aliases": ["Srivatsava", "Srivastava"],
        "affil": "Stanford",
    },
    {
        "name": "Daneshvar Amrollahi",
        "emails": ["daneshvar@cs.stanford.edu"],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Elyas Obbad",
        "emails": [
            "eobbad@skampere1.stanford.edu",
            "eobbad@ampere1.stanford.edu",
        ],
        "github": "eobbad",
        "aliases": ["Elyas", "eobbad"],
        "affil": "Stanford",
    },
    {
        "name": "Iddah Mlauzi",
        "emails": [
            "iddah@stanford.edu",
            "iddah@mercury1.stanford.edu",
            "iddah@ampere1.stanford.edu",
        ],
        "github": "iddahmlauzi",
        "aliases": ["Iddah", "Iddah Ashley Kudakwashe Mlauzi", "iddahmlauzi"],
        "affil": "Stanford",
    },
    {
        "name": "Weston Kirk",
        "emails": [
            "wkirk@stanford.edu",
            "wkirk@skampere1.stanford.edu",
        ],
        "github": "wkirk11",
        "aliases": ["wkirk11", "Weston Kirk"],
        "affil": "Stanford",
    },
    {
        "name": "Slim Barkallah",
        "emails": ["slimbark@skampere1.stanford.edu"],
        "github": "Slim205",
        "aliases": ["Slim BARKALLAH", "Slim Barkallah"],
        "affil": "Stanford",
    },
    {
        "name": "Shurui Liu",
        "emails": ["srliu3264@gmail.com"],
        "github": None,
        "aliases": ["Shurui"],
        "affil": "Stanford",
    },
    {
        "name": "Louise Li",
        "emails": [
            "louisely@skampere1.stanford.edu",
            "liying_louisechina@163.com",
        ],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Saumya Goyal",
        "emails": ["saumyagoyal01@gmail.com", "saumyag@andrew.cmu.edu"],
        "github": None,
        "aliases": ["Saumya", "saumyag"],
        "affil": "CMU",
    },
    {
        "name": "Rakshit Kaushik",
        "emails": ["rakshit.kaushik2772@gmail.com"],
        "github": "rakshit-kaushik",
        "aliases": ["rakshit-kaushik"],
        "affil": None,
    },
    {
        "name": "Mahyar Karimi",
        "emails": ["mahyar.karimi2079@gmail.com"],
        "github": "mahykari",
        "aliases": ["mahykari"],
        "affil": None,
    },
    {
        "name": "Yegor Denisov-Blanch",
        "emails": ["ydenisov8@gmail.com"],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Krrish Chawla",
        "emails": ["krrish@ampere1.stanford.edu"],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Kirill Acharya",
        "emails": ["kirillacharya@formore.local"],
        "github": None,
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Andrea Yu",
        "emails": [],
        "github": "aandreayu",
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Andrew Zhou",
        "emails": ["andrewzhou924@gmail.com"],
        "github": "andrewzhou924",
        "aliases": ["AndrewZhou924"],
        "affil": None,
    },
    {
        "name": "Amy Lu",
        "emails": ["amyluaglaia@gmail.com"],
        "github": "AmyLu0828",
        "aliases": ["AmyLu0828"],
        "affil": None,
    },
    {
        "name": "Aryan Gulati",
        "emails": ["aryanguls@gmail.com"],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Alycia Lee",
        "emails": ["alycialee@microsoft.com"],
        "github": None,
        "aliases": [],
        "affil": "Microsoft",
    },
    {
        "name": "Alex Shaw",
        "emails": ["alexgshaw64@gmail.com"],
        "github": None,
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Belinda Mo",
        "emails": ["justanexperimentpark@gmail.com"],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Brian Ko",
        "emails": ["briankosw@gmail.com"],
        "github": "briankosw",
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Ching-Tsun Chou",
        "emails": ["chingtsun.chou@gmail.com"],
        "github": None,
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Emily Xia",
        "emails": [
            "emxia18@gmail.com",
            "emxia18@hyperturing1.stanford.edu",
        ],
        "github": "emxia18",
        "aliases": ["emxia18"],
        "affil": "Stanford",
    },
    {
        "name": "Eshaan Barkataki",
        "emails": ["eshaanbarkataki@gmail.com"],
        "github": "Eshaancoding",
        "aliases": ["Eshaancoding"],
        "affil": None,
    },
    {
        "name": "Henry Bosch",
        "emails": ["henrybosch15@gmail.com"],
        "github": None,
        "aliases": ["henry"],
        "affil": None,
    },
    {
        "name": "Hongzhou Lin",
        "emails": ["lhongzho@amazon.com"],
        "github": None,
        "aliases": [],
        "affil": "Amazon",
    },
    {
        "name": "Jize (Winston) Jiang",
        "emails": ["jiangjize2000@gmail.com"],
        "github": "L-Penguin",
        "aliases": ["L-Penguin"],
        "affil": None,
    },
    {
        "name": "Matt Chen",
        "emails": ["mattchen05@gmail.com"],
        "github": "cymcymcymcym",
        "aliases": ["cymcymcymcym"],
        "affil": None,
    },
    {
        "name": "Nathan Fulton",
        "emails": ["gitcommit@nfulton.org"],
        "github": None,
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Ethan Li",
        "emails": ["ethanlnow@gmail.com"],
        "github": "lethan3",
        "aliases": ["lethan3"],
        "affil": None,
    },
    {
        "name": "Stepan",
        "emails": ["stepurik@stanford.edu"],
        "github": None,
        "aliases": [],
        "affil": "Stanford",
    },
    {
        "name": "Vasily Pestun",
        "emails": ["vasily.pestun@gmail.com"],
        "github": None,
        "aliases": [],
        "affil": None,
    },
    {
        "name": "Allen Nie",
        "emails": ["leo.niecn@gmail.com"],
        "github": "windweller",
        "aliases": ["windweller"],
        "affil": "Stanford",
    },
    {
        "name": "Mr-aio",
        "emails": [],
        "github": "Mr-aio",
        "aliases": ["Mr-aio"],
        "affil": None,
    },
]


# ---------------------------------------------------------------------------
# Derived lookup tables
# ---------------------------------------------------------------------------

by_email: dict[str, dict[str, Any]] = {
    e: c for c in COLLABORATORS for e in c.get("emails", [])
}
by_name: dict[str, dict[str, Any]] = {c["name"]: c for c in COLLABORATORS}
by_github: dict[str, dict[str, Any]] = {
    c["github"]: c for c in COLLABORATORS if c.get("github")
}
by_alias: dict[str, dict[str, Any]] = {
    a: c for c in COLLABORATORS for a in c.get("aliases", [])
}


def resolve(key: str) -> dict[str, Any] | None:
    """Resolve a name / email / github handle / alias to a collaborator record.

    Tries (in order): by_email, by_github, by_name, by_alias.
    Case-sensitive for emails and github; case-insensitive fallback for names.
    """
    if key in by_email:
        return by_email[key]
    if key in by_github:
        return by_github[key]
    if key in by_name:
        return by_name[key]
    if key in by_alias:
        return by_alias[key]
    low = key.lower()
    for name, rec in by_name.items():
        if name.lower() == low:
            return rec
    for alias, rec in by_alias.items():
        if alias.lower() == low:
            return rec
    return None


__all__ = [
    "COLLABORATORS",
    "by_email",
    "by_name",
    "by_github",
    "by_alias",
    "resolve",
]
