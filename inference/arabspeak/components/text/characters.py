from typing import List, Dict

class Characters:
    def __init__(
        self,
        char_to_id: Dict[str, int],
        pad: str,
        eos: str,
        bos: str,
        blank: str,
    ):
        self._char_to_id = char_to_id.copy()
        self._id_to_char = {id: char for char, id in char_to_id.items()}
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self._num_chars = len(char_to_id)
        
        # Validate that special tokens are in the mappings
        self._validate_special_tokens(pad, eos, bos, blank)
        
    def _validate_special_tokens(self, pad: str, eos: str, bos: str, blank: str):
        special_tokens = [pad, eos, bos, blank]
        for token in special_tokens:
            if token not in self._char_to_id:
                raise ValueError(f"Special token '{token}' is not in char_to_id mapping.")
    
    @property
    def pad_id(self) -> int:
        return self._char_to_id[self._pad]
    
    @property
    def eos_id(self) -> int:
        return self._char_to_id[self._eos]
    
    @property
    def bos_id(self) -> int:
        return self._char_to_id[self._bos]
    
    @property
    def blank_id(self) -> int:
        return self._char_to_id[self._blank]
    
    @property
    def num_chars(self) -> int:
        return self._num_chars
    
    def char_to_id(self, char: str) -> int:
        try:
            return self._char_to_id[char]
        except KeyError:
            raise KeyError(f"Character '{char}' not found in character mapping.")
    
    def id_to_char(self, idx: int) -> str:
        try:
            return self._id_to_char[idx]
        except KeyError:
            raise KeyError(f"Index '{idx}' not found in character mapping.")

    def chars_to_ids(self, chars: List[str]) -> List[int]:
        return [self.char_to_id(char) for char in chars]

    def ids_to_chars(self, ids: List[int]) -> List[str]:
        return [self.id_to_char(idx) for idx in ids]
    
    @staticmethod
    def init_from_config(config: Dict) -> 'Characters':
        required_keys = ['char_to_id', 'pad', 'eos', 'bos', 'blank']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Config missing required key: {key}")
        return Characters(
            char_to_id=config['char_to_id'],
            pad=config['pad'],
            eos=config['eos'],
            bos=config['bos'],
            blank=config['blank'],
        )
    
    def to_config(self) -> Dict:
        return {
            'char_to_id': self._char_to_id,
            'pad': self._pad,
            'eos': self._eos,
            'bos': self._bos,
            'blank': self._blank
        }