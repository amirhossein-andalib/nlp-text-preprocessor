from preprocessor import PersianPreprocessor, EnglishPreprocessor


def main():
    print("=" * 60)
    print("   NLP TEXT PREPROCESSOR — Persian & English")
    print("=" * 60)

    # ── Persian Example ──
    persian_text = "زبان‌شناسی علمی است که به مطالعه‌ی زبان‌های انسانی می‌پردازد."
    persian_prep = PersianPreprocessor()
    persian_result = persian_prep.preprocess(persian_text, do_lemmatize=True)

    # ── English Example ──
    english_text = "Natural Language Processing is a fascinating field of Artificial Intelligence!"
    english_prep = EnglishPreprocessor()
    english_result = english_prep.preprocess(english_text, do_lemmatize=True)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Persian tokens : {persian_result}")
    print(f"English tokens : {english_result}")


if __name__ == "__main__":
    main()
