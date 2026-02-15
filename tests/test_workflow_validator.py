"""Tests for workflow field validation."""

from server.engine.workflow_executor import validate_field


def test_phone_valid():
    field = {"label": "电话", "field_type": "phone", "required": True}
    assert validate_field("13812345678", field) is None


def test_phone_invalid():
    field = {"label": "电话", "field_type": "phone", "required": True}
    assert validate_field("12345", field) is not None


def test_id_card_valid():
    field = {"label": "身份证", "field_type": "id_card", "required": True}
    assert validate_field("110101199001011234", field) is None
    assert validate_field("11010119900101123X", field) is None


def test_id_card_invalid():
    field = {"label": "身份证", "field_type": "id_card", "required": True}
    assert validate_field("1234", field) is not None


def test_required_empty():
    field = {"label": "姓名", "field_type": "text", "required": True}
    assert validate_field("", field) is not None


def test_optional_empty():
    field = {"label": "备注", "field_type": "text", "required": False}
    assert validate_field("", field) is None


def test_custom_regex():
    field = {"label": "邮编", "field_type": "text", "required": True, "validation_rule": r"^\d{6}$"}
    assert validate_field("310000", field) is None
    assert validate_field("31000", field) is not None


def test_email_valid():
    field = {"label": "邮箱", "field_type": "email", "required": True}
    assert validate_field("test@example.com", field) is None


def test_email_invalid():
    field = {"label": "邮箱", "field_type": "email", "required": True}
    assert validate_field("not-an-email", field) is not None
