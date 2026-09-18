from string import Formatter

import pytest

from storepilot.config import StrategyConfig
from storepilot.i18n import EN, ValidationError, translate
from storepilot.workbench import prepare_review


def test_translations_preserve_format_arguments():
    formatter = Formatter()
    for source, translated in EN.items():
        source_fields = {field for _, field, _, _ in formatter.parse(source) if field is not None}
        translated_fields = {
            field for _, field, _, _ in formatter.parse(translated) if field is not None
        }
        assert translated_fields == source_fields, source


def test_validation_errors_keep_default_message_and_render_in_english():
    with pytest.raises(ValueError) as caught:
        StrategyConfig(review_period_days=0).validate()
    assert str(caught.value) == "盘点周期必须为正整数"
    assert isinstance(caught.value, ValidationError)
    assert caught.value.render("en") == "The review period must be a positive integer."


def test_order_error_preserves_the_case_pack_and_minimum():
    with pytest.raises(ValidationError) as caught:
        prepare_review(
            {"suggested_order_qty": 12, "case_pack": 6, "min_order_qty": 12},
            "修改",
            7,
            "库存复核",
            "",
        )
    assert "6" in caught.value.render("en")
    assert "minimum of 12" in caught.value.render("en")
    assert "起订量为 12" in str(caught.value)


def test_unknown_text_is_preserved():
    assert translate("Customer's own product {name}", "en") == "Customer's own product {name}"
