from typing import Callable, Dict, List, Union
from .phonemizer import Phonemizer
from .characters import Characters
from .cleaners import clean_text


class Tokenizer:
    def __init__(
        self,
        text_cleaner: Callable = None,
        characters: "Characters" = None,
        phonemizer: "Phonemizer" = None,
        add_blank: bool = False,
        use_eos_bos: bool=False,
    ):
        self.text_cleaner = text_cleaner
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.characters = characters
        self.not_found_characters = []
        self.phonemizer = phonemizer

    def encode(self, text: str) -> List[int]:  # pylint: disable=unused-argument
        """Converts a string of text to a sequence of token IDs as follows:
        1. Text normalizatin
        2. Phonemization 
        3. Add blank char between characters if specified
        4. Add BOS and EOS characters if specified
        5. Text to token IDs

        Args:
            text(str):
                The text to convert to token IDs.
        returns:
            token_ids(List[int]):
                The token IDs of the text.
        """
        if self.text_cleaner is not None:
            text = self.text_cleaner(text)
        text = self.phonemizer.phonemize(text)
        text = self.text_to_ids(text)
        if self.add_blank:
            text = self.intersperse_blank_char(text, True)
        if self.use_eos_bos:
            text = self.pad_with_bos_eos(text)
        return text

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        phonems = ""
        for token_id in token_ids:
            try:
                if token_id in [self.characters.blank_id, self.characters.bos_id, self.characters.eos_id]:
                    continue
                phonems += self.characters.id_to_char(token_id)
            except KeyError:
                print(f" [!] Token ID {token_id} not found in the vocabulary. Discarding it.")
        phonems = phonems.replace(self.phonemizer.separator, "")
        return self.phonemizer.buckwalter_to_arabic(phonems)

    def text_to_ids(self, text: str) -> List[int]:
        """Encodes a string of text to a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    print(text)
                    print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos_id] + list(char_sequence) + [self.characters.eos_id]

    def intersperse_blank_char(self, char_sequence: List[str]):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank_id 
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

    @staticmethod
    def init_from_config(config: dict):
        """Init Tokenizer object from config

        Args:
            config: Tokenizer config.
        """
        required_keys = ["characters","phonemizer"]
        optional_keys = ["add_blank", "use_eos_bos"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Config missing required key: {key}")

        for key in optional_keys:
            if key not in config:
                config[key] = False
        phonemizer = Phonemizer.init_from_config(config["phonemizer"])
        characters = Characters.init_from_config(config["characters"])
        addd_blank = config["add_blank"]
        use_eos_bos = config["use_eos_bos"]

        return Tokenizer(
                phonemizer=phonemizer,
                characters=characters,
                add_blank=addd_blank,
                use_eos_bos=use_eos_bos,
                text_cleaner=clean_text,
        )
