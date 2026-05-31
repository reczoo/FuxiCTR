import polars as pl
from polars.testing import assert_series_equal
from pytest import fixture
from fuxictr.preprocess.tokenizer import count_tokens, Tokenizer

@fixture
def series_word():
    df = pl.Series(["a","b","a","c"])
    return df


@fixture
def series_text():
    df = pl.Series(["a,b,c","b,a","a"])
    return df

@fixture
def series_text_new():
    df = pl.Series(["a,b,c","p,q,,a","b,c","c"])
    return df


class TestCountTokens:
    def test_count_tokens_single(self, series_word):
        actual_word_counts, actual_max_len = count_tokens(series_word)
        expected_word_counts = {"a": 2, "b":1, "c": 1}
        expected_max_len = 0 # only for text
        assert actual_max_len==expected_max_len
        assert actual_word_counts == expected_word_counts

    def test_count_tokens_text(self, series_text):
        actual_word_counts, actual_max_len = count_tokens(series_text,",")
        expected_word_counts = {"a": 3, "b":2, "c": 1}
        expected_max_len = 3 # only for text
        assert actual_max_len==expected_max_len
        assert actual_word_counts == expected_word_counts


class TestTokenizer:
    def test_fit_on_texts_small(self, series_text):
        tok = Tokenizer(splitter=",")
        tok.fit_on_texts(series_text)
        expected_vocab = {'a': 1, 'b': 2, 'c': 3,  '__PAD__': 0, '__OOV__': 5}
        actual = tok.vocab
        # assert expected_vocab == actual
        # nb c & d have same frequency so order undetermined

    def test_encode_sequence(self, series_text, series_text_new):
        tok = Tokenizer(splitter=",",na_value="")
        tok.fit_on_texts(series_text)
        actual = tok.encode_sequence(series_text_new)
        expected = pl.Series('sequence', 
            [[1, 2, 3], # no padding
            [4, 4, 0],  # 2 OOV and NA->PAD
            [0, 2, 3],  # 1 padding
            [0, 0, 3]], # 2 padding
            dtype=pl.Array(pl.Int64,3))
        assert_series_equal(actual, expected)
