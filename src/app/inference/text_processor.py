from cleantext import clean


class TextProcessor:    
    @staticmethod
    def clean_text(text: str) -> str:
        return clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            replace_with_punct="",
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            lang="en"
        )
