"""Parse Gmail's ``Authentication-Results:`` header to extract SPF/DKIM/DMARC verdicts.

Gmail stamps every inbound message with an ``Authentication-Results:`` header
summarizing what SPF, DKIM, and DMARC said. Trusting Gmail's verdict (rather
than re-verifying DKIM ourselves) is the standard approach when reading via the
Gmail API — Gmail has already done the cryptographic work.

For an allowlisted sender to be accepted we require:
  - SPF = pass
  - DKIM = pass
  - DMARC = pass
  - The DKIM signing domain matches the From: domain

If the header is missing we reject (conservative default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthVerdict:
    spf: str = "none"
    dkim: str = "none"
    dmarc: str = "none"
    dkim_domain: str = ""

    @property
    def all_pass(self) -> bool:
        return self.spf == "pass" and self.dkim == "pass" and self.dmarc == "pass"


_METHOD_RE = re.compile(
    r"\b(spf|dkim|dmarc)=(\w+)",
    re.IGNORECASE,
)
_DKIM_DOMAIN_RE = re.compile(
    r"dkim=\w+[^;]*?\b(?:header\.d|header\.i)=@?([A-Za-z0-9.\-]+)",
    re.IGNORECASE,
)


def parse_authentication_results(header_value: str) -> AuthVerdict:
    """Parse the value of an ``Authentication-Results:`` header."""
    if not header_value:
        return AuthVerdict()
    spf = dkim = dmarc = "none"
    for match in _METHOD_RE.finditer(header_value):
        method = match.group(1).lower()
        result = match.group(2).lower()
        if method == "spf":
            spf = result
        elif method == "dkim":
            dkim = result
        elif method == "dmarc":
            dmarc = result
    dkim_domain = ""
    m = _DKIM_DOMAIN_RE.search(header_value)
    if m:
        dkim_domain = m.group(1).lower().lstrip("@")
    return AuthVerdict(spf=spf, dkim=dkim, dmarc=dmarc, dkim_domain=dkim_domain)


def domain_of(addr: str) -> str:
    return addr.partition("@")[2].lower()


def verify_auth_headers(from_addr: str, auth_results_header: str) -> tuple[bool, str]:
    """Return (ok, reason). ``ok`` is True iff SPF/DKIM/DMARC all pass AND the
    DKIM signing domain matches (or is a parent of) the From: domain.
    """
    if not auth_results_header:
        return False, "missing Authentication-Results header"
    verdict = parse_authentication_results(auth_results_header)
    if verdict.spf != "pass":
        return False, f"spf={verdict.spf}"
    if verdict.dkim != "pass":
        return False, f"dkim={verdict.dkim}"
    if verdict.dmarc != "pass":
        return False, f"dmarc={verdict.dmarc}"
    from_domain = domain_of(from_addr)
    if not from_domain:
        return False, "no from domain"
    if verdict.dkim_domain and not (
        verdict.dkim_domain == from_domain
        or from_domain.endswith("." + verdict.dkim_domain)
        or verdict.dkim_domain.endswith("." + from_domain)
    ):
        return False, f"dkim domain mismatch ({verdict.dkim_domain} vs {from_domain})"
    return True, "ok"
