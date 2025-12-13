import pytest

from src.app.inference.text_processor import TextProcessor


class TestTextProcessor:
    def test_clean_text_basic(self):
        processor = TextProcessor()
        text = "Hello World!"
        result = processor.clean_text(text)
        assert result == "hello world!"

    def test_clean_text_with_url(self):
        processor = TextProcessor()
        text = "Check this out https://example.com"
        result = processor.clean_text(text)
        assert "https://example.com" not in result
        assert "check this out" in result

    def test_clean_text_with_email(self):
        processor = TextProcessor()
        text = "Contact me at test@example.com"
        result = processor.clean_text(text)
        assert "test@example.com" not in result
        assert "contact me at" in result

    def test_clean_text_lowercase(self):
        processor = TextProcessor()
        text = "UPPERCASE TEXT"
        result = processor.clean_text(text)
        assert result == "uppercase text"

    def test_clean_text_empty_string(self):
        processor = TextProcessor()
        text = ""
        result = processor.clean_text(text)
        assert result == ""

    def test_clean_text_with_numbers(self):
        processor = TextProcessor()
        text = "I have 42 apples"
        result = processor.clean_text(text)
        assert "42" in result
