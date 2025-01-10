import pytest
from inference.arabspeak.components.tokenizer import TTSTokenizer
from inference.arabspeak.components.characters import Characters
from inference.arabspeak.components.phonemizer import IPAPhonemizer

@pytest.fixture
def characters():
    # Define a simple character set for testing
    return Characters(
        pad="<pad>",
        bos="<s>",
        eos="</s>",
        blank="<blank>",
        chars_list=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"],
    )

@pytest.fixture
def phonemizer():
    # Define a simple phonemizer for testing
    return IPAPhonemizer()

@pytest.fixture
def tokenizer(characters, phonemizer):
    # Create a tokenizer instance with the test character set
    return TTSTokenizer(characters=characters, phonemizer=phonemizer)

def test_encode(tokenizer, characters):
    # Test encoding a simple string
    text = "abc"
    expected_ids = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c")]
    assert tokenizer.encode(text) == expected_ids

def test_encode_with_unknown_characters(tokenizer, characters):
    # Test encoding a string with unknown characters
    text = "abz123"
    expected_ids = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("z")]
    assert tokenizer.encode(text) == expected_ids
    assert tokenizer.not_found_characters == ["1", "2", "3"]

def test_decode(tokenizer, characters):
    # Test decoding a list of IDs
    ids = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c")]
    expected_text = "abc"
    assert tokenizer.decode(ids) == expected_text

def test_text_to_ids(tokenizer, characters):
    # Test text_to_ids method without blank or bos/eos
    text = "abc"
    expected_ids = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c")]
    assert tokenizer.text_to_ids(text) == expected_ids

def test_text_to_ids_with_blank(characters, phonemizer):
    # Test text_to_ids method with blank
    tokenizer = TTSTokenizer(characters=characters, phonemizer=phonemizer, add_blank=True)
    text = "abc"
    expected_ids = [
        characters.blank_id,
        characters.char_to_id("a"),
        characters.blank_id,
        characters.char_to_id("b"),
        characters.blank_id,
        characters.char_to_id("c"),
        characters.blank_id,
    ]
    assert tokenizer.text_to_ids(text) == expected_ids

def test_text_to_ids_with_bos_eos(characters, phonemizer):
    # Test text_to_ids method with bos/eos
    tokenizer = TTSTokenizer(characters=characters, phonemizer=phonemizer, use_eos_bos=True)
    text = "abc"
    expected_ids = [
        characters.bos_id,
        characters.char_to_id("a"),
        characters.char_to_id("b"),
        characters.char_to_id("c"),
        characters.eos_id,
    ]
    assert tokenizer.text_to_ids(text) == expected_ids

def test_text_to_ids_with_blank_and_bos_eos(characters, phonemizer):
    # Test text_to_ids method with blank and bos/eos
    tokenizer = TTSTokenizer(characters=characters, phonemizer=phonemizer, add_blank=True, use_eos_bos=True)
    text = "abc"
    expected_ids = [
        characters.bos_id,
        characters.blank_id,
        characters.char_to_id("a"),
        characters.blank_id,
        characters.char_to_id("b"),
        characters.blank_id,
        characters.char_to_id("c"),
        characters.blank_id,
        characters.eos_id,
    ]
    assert tokenizer.text_to_ids(text) == expected_ids

def test_ids_to_text(tokenizer, characters):
    # Test ids_to_text method
    ids = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c")]
    expected_text = "abc"
    assert tokenizer.ids_to_text(ids) == expected_text

def test_pad_with_bos_eos(tokenizer, characters):
    # Test pad_with_bos_eos method
    char_sequence = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c")]
    expected_sequence = [characters.bos_id, characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c"), characters.eos_id]
    assert tokenizer.pad_with_bos_eos(char_sequence) == expected_sequence

def test_intersperse_blank_char(tokenizer, characters):
    # Test intersperse_blank_char method
    char_sequence = [characters.char_to_id("a"), characters.char_to_id("b"), characters.char_to_id("c")]
    expected_sequence = [
        characters.blank_id,
        characters.char_to_id("a"),
        characters.blank_id,
        characters.char_to_id("b"),
        characters.blank_id,
        characters.char_to_id("c"),
        characters.blank_id,
    ]
    assert tokenizer.intersperse_blank_char(char_sequence) == expected_sequence
