from src.auth_headers import parse_authentication_results, verify_auth_headers


GOOD = (
    "mx.google.com; "
    "dkim=pass header.i=@gmail.com header.s=20230601 header.b=abc; "
    "spf=pass (google.com: domain of brando.science@gmail.com designates 1.2.3.4) "
    "smtp.mailfrom=brando.science@gmail.com; "
    "dmarc=pass (p=NONE sp=QUARANTINE dis=NONE) header.from=gmail.com"
)

SPF_FAIL = GOOD.replace("spf=pass", "spf=fail")
DKIM_FAIL = GOOD.replace("dkim=pass", "dkim=fail")
DMARC_FAIL = GOOD.replace("dmarc=pass", "dmarc=fail")
WRONG_DKIM_DOMAIN = GOOD.replace("header.i=@gmail.com", "header.i=@evil.example")

STANFORD_GOOD = (
    "mx.google.com; "
    "dkim=pass header.i=@stanford.edu header.s=stanford header.b=xyz; "
    "spf=pass smtp.mailfrom=brando9@stanford.edu; "
    "dmarc=pass header.from=stanford.edu"
)


class TestParse:
    def test_all_pass(self):
        v = parse_authentication_results(GOOD)
        assert v.spf == "pass"
        assert v.dkim == "pass"
        assert v.dmarc == "pass"
        assert v.dkim_domain == "gmail.com"
        assert v.all_pass

    def test_empty_returns_none_defaults(self):
        v = parse_authentication_results("")
        assert v.spf == "none"
        assert v.dkim == "none"
        assert v.dmarc == "none"
        assert not v.all_pass


class TestVerifyAuthHeaders:
    def test_good_gmail(self):
        ok, reason = verify_auth_headers("brando.science@gmail.com", GOOD)
        assert ok, reason

    def test_good_stanford(self):
        ok, reason = verify_auth_headers("brando9@stanford.edu", STANFORD_GOOD)
        assert ok, reason

    def test_spf_fail(self):
        ok, reason = verify_auth_headers("brando.science@gmail.com", SPF_FAIL)
        assert not ok
        assert "spf" in reason

    def test_dkim_fail(self):
        ok, reason = verify_auth_headers("brando.science@gmail.com", DKIM_FAIL)
        assert not ok
        assert "dkim" in reason

    def test_dmarc_fail(self):
        ok, reason = verify_auth_headers("brando.science@gmail.com", DMARC_FAIL)
        assert not ok
        assert "dmarc" in reason

    def test_missing_header(self):
        ok, reason = verify_auth_headers("brando.science@gmail.com", "")
        assert not ok
        assert "missing" in reason.lower()

    def test_dkim_domain_mismatch(self):
        ok, reason = verify_auth_headers("brando.science@gmail.com", WRONG_DKIM_DOMAIN)
        assert not ok
        assert "dkim" in reason.lower()
