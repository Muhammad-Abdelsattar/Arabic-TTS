from typing import Dict, List

class Phonemizer:
    def __init__(self,
                 char_to_phonem: Dict[str, str],):

        self._char_to_phonem = char_to_phonem.copy()
        self._phonem_to_char = {v: k for k, v in char_to_phonem.items()}

    def arabic_to_buckwalter(self, text: str) -> str:
        """
        Converts Arabic text to Buckwalter transliteration.
        args:
            text (str): The Arabic text to be converted.
        returns:
            str: The Buckwalter transliteration of the input text.
        """
        buckwalter_text = ""
        for char in text:
            if char in self._char_to_phonem:
                buckwalter_text += self._char_to_phonem[char]
            else:
                buckwalter_text += char
        return buckwalter_text
    
    def buckwalter_to_arabic(self, text: str) -> str:
        """
        Converts Buckwalter transliteration to Arabic text.
        args:
            text (str): The Buckwalter transliteration to be converted.
        returns:
            str: The Arabic text corresponding to the input Buckwalter transliteration.
        """
        arabic_text = ""
        for char in text:
            if char in self._phonem_to_char:
                arabic_text += self._phonem_to_char[char]
            else:
                arabic_text += char
        return arabic_text

    def phonemize(self, text: str) -> str:
        """
        Phonemizes a string of text.
        args:
            text (str): The text to be phonemized.
        returns:
            str: A string of phonemes corresponding to the input text.
        """
        phonemized_text = self.arabic_to_buckwalter(text)
        return phonemized_text

    def phonemize_batch(self, texts: List[str]) -> List[str]:
        """
        Phonemizes a list of strings of text.
        args:
            texts (List[str]): A list of strings of text to be phonemized.
        returns:
            List[List[str]]: A list of lists of phonemes corresponding to the input texts.
        """
        phonemized_texts = [self.phonemize(text) for text in texts]
        return phonemized_texts

    @staticmethod
    def init_from_config(config: Dict) -> "Phonemizer":
        """
        Initializes a Phonemizer object from a configuration dictionary.
        args:
            config (Dict): A dictionary containing the configuration parameters for the Phonemizer.
        returns:
            Phonemizer: A Phonemizer object initialized with the provided configuration.
        """
        required_keys = ['char_to_phonem']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Config missing required key: {key}")
        char_to_phonem = config["char_to_phonem"]
        return Phonemizer(char_to_phonem)

    @staticmethod
    def to_config(self) -> Dict:
        """
        Converts the Phonemizer object to a configuration dictionary.
        returns:
            Dict: A dictionary containing the configuration parameters for the Phonemizer.
        """
        return {
            'char_to_phonem': self._char_to_phonem,
        }