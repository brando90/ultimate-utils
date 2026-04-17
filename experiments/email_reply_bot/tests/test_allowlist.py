from src.allowlist import normalize_addr, verify_sender


class TestNormalize:
    def test_plain(self):
        assert normalize_addr("brando9@stanford.edu") == "brando9@stanford.edu"

    def test_uppercase(self):
        assert normalize_addr("Brando.Science@Gmail.com") == "brando.science@gmail.com"

    def test_display_name(self):
        assert normalize_addr('"Brando" <brandojazz@gmail.com>') == "brandojazz@gmail.com"

    def test_plus_tag(self):
        assert normalize_addr("brando.science+reports@gmail.com") == "brando.science@gmail.com"

    def test_plus_tag_with_display(self):
        assert normalize_addr('Brando <brando.science+x@gmail.com>') == "brando.science@gmail.com"

    def test_empty(self):
        assert normalize_addr("") == ""

    def test_garbage(self):
        assert normalize_addr("not-an-email") == ""

    def test_whitespace(self):
        assert normalize_addr("  brando9@stanford.edu  ") == "brando9@stanford.edu"


class TestVerify:
    def test_all_three_allowed(self):
        assert verify_sender("brando.science@gmail.com")
        assert verify_sender("brandojazz@gmail.com")
        assert verify_sender("brando9@stanford.edu")

    def test_alias_accepted(self):
        assert verify_sender("brando.science+nightly@gmail.com")

    def test_case_insensitive(self):
        assert verify_sender("BRANDO9@stanford.edu")

    def test_with_display_name(self):
        assert verify_sender('"Brando Miranda" <brando.science@gmail.com>')

    def test_stranger_rejected(self):
        assert not verify_sender("random@example.com")

    def test_similar_stranger_rejected(self):
        assert not verify_sender("brando9@gmail.com")
        assert not verify_sender("brando.science@example.com")
        assert not verify_sender("brandojazz@stanford.edu")

    def test_empty_rejected(self):
        assert not verify_sender("")

    def test_no_domain_rejected(self):
        assert not verify_sender("brando9")
