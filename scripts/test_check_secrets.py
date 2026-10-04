# Fake keys are assembled at runtime so this file itself never contains a
# string the scanner (or GitHub push protection) would flag.
import check_secrets as cs


def _names(text: str) -> set[str]:
    return {name for _, name, _ in cs.scan_text(text)}


def test_detects_anthropic_key() -> None:
    assert "anthropic_key" in _names("KEY=" + "sk-ant-" + "api03-" + "a1B2" * 10)


def test_detects_openai_project_key() -> None:
    assert "openai_key" in _names("x = '" + "sk-" + "proj-" + "Z9" * 24 + "'")


def test_detects_aws_and_github_and_private_key() -> None:
    text = "\n".join([
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "ghp_" + "a" * 36,
        "-----BEGIN RSA " + "PRIVATE KEY-----",
    ])
    assert {"aws_access_key", "github_token", "private_key"} <= _names(text)


def test_generic_assignment_detected_but_placeholders_pass() -> None:
    assert "generic_secret" in _names('api_key = "' + "q8Zt" * 8 + '"')
    assert _names(open(".env.example").read()) == set()


def test_allow_comment_suppresses() -> None:
    assert _names("sk-ant-" + "b" * 30 + "  # secret-scan: allow") == set()


def test_env_file_is_forbidden_by_name_but_example_is_not() -> None:
    assert cs.scan_path(".env", "")
    assert cs.scan_path("api/.env.local", "")
    assert cs.scan_path(".env.example", "") == []


def test_output_masks_the_secret() -> None:
    key = "sk-ant-" + "c" * 40
    (_, _, masked), = cs.scan_text(key)
    assert key not in masked
