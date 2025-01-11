import os
import json

arabic_to_buckwalter = { #mapping from Arabic script to Buckwalter
	u'\u0628': u'b' , u'\u0630': u'*' , u'\u0637': u'T' , u'\u0645': u'm',
	u'\u062a': u't' , u'\u0631': u'r' , u'\u0638': u'Z' , u'\u0646': u'n',
	u'\u062b': u'^' , u'\u0632': u'z' , u'\u0639': u'E' , u'\u0647': u'h',
	u'\u062c': u'j' , u'\u0633': u's' , u'\u063a': u'g' , u'\u062d': u'H',
	u'\u0642': u'q' , u'\u0641': u'f' , u'\u062e': u'x' , u'\u0635': u'S',
	u'\u0634': u'$' , u'\u062f': u'd' , u'\u0636': u'D' , u'\u0643': u'k',
	u'\u0623': u'>' , u'\u0621': u'\'', u'\u0626': u'}' , u'\u0624': u'&',
	u'\u0625': u'<' , u'\u0622': u'|' , u'\u0627': u'A' , u'\u0649': u'Y',
	u'\u0629': u'p' , u'\u064a': u'y' , u'\u0644': u'l' , u'\u0648': u'w',
	u'\u064b': u'F' , u'\u064c': u'N' , u'\u064d': u'K' , u'\u064e': u'a',
	u'\u064f': u'u' , u'\u0650': u'i' , u'\u0651': u'~' , u'\u0652': u'o'
}

buckwalter_to_arabic = { #mapping from Buckwalter to Arabic script
	u'b': u'\u0628' , u'*': u'\u0630' , u'T': u'\u0637' , u'm': u'\u0645',
	u't': u'\u062a' , u'r': u'\u0631' , u'Z': u'\u0638' , u'n': u'\u0646',
	u'^': u'\u062b' , u'z': u'\u0632' , u'E': u'\u0639' , u'h': u'\u0647',
	u'j': u'\u062c' , u's': u'\u0633' , u'g': u'\u063a' , u'H': u'\u062d',
	u'q': u'\u0642' , u'f': u'\u0641' , u'x': u'\u062e' , u'S': u'\u0635',
	u'$': u'\u0634' , u'd': u'\u062f' , u'D': u'\u0636' , u'k': u'\u0643',
	u'>': u'\u0623' , u'\'': u'\u0621', u'}': u'\u0626' , u'&': u'\u0624',
	u'<': u'\u0625' , u'|': u'\u0622' , u'A': u'\u0627' , u'Y': u'\u0649',
	u'p': u'\u0629' , u'y': u'\u064a' , u'l': u'\u0644' , u'w': u'\u0648',
	u'F': u'\u064b' , u'N': u'\u064c' , u'K': u'\u064d' , u'a': u'\u064e',
	u'u': u'\u064f' , u'i': u'\u0650' , u'~': u'\u0651' , u'o': u'\u0652'
}

def get_phonemizer_config():
    config = {}
    config["char_to_phonem"] = arabic_to_buckwalter
    return config

def get_character_mapping(model):
    return model.tokenizer.characters._char_to_id

def get_characters_config(model):
    config = {}
    config["characters"] = model.tokenizer.characters.characters
    config["punctuations"] = model.tokenizer.characters.punctuations
    config["pad"] = model.tokenizer.characters.pad
    config["bos"] = model.tokenizer.characters.bos
    config["eos"] = model.tokenizer.characters.eos
    config["blank"] = model.tokenizer.characters.blank
    config["char_to_id"] = get_character_mapping(model)
    return config


def get_tokenizer_config(model):
    config = {}
    config["add_blank"] = model.tokenizer.add_blank
    config["use_eos_bos"] = model.tokenizer.use_eos_bos
    config["phonemizer"] = get_phonemizer_config()
    config["characters"] = get_characters_config(model)
    return config

def save_tokenizer_config(path,config):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)

def get_latest_checkpoint(directory):
    """
    Finds the latest checkpoint file in the specified directory.

    Parameters:
    directory (str): The path to the directory containing the checkpoints.

    Returns:
    str: The filename of the latest checkpoint file, or None if no checkpoint files are found.
    """
    latest_checkpoint = None
    highest_number = -1

    # List all files in the directory
    files = os.listdir(directory)

    # Iterate through the files
    for file in files:
        # Check if the file matches the pattern checkpoint_{number}.pth
        if file.startswith("checkpoint_") and file.endswith(".pth"):
            # Extract the number from the filename
            parts = file.split("_")
            if len(parts) >= 2:
                num_str = parts[1].split(".")[0]
                if num_str.isdigit():
                    number = int(num_str)
                    # Update if this is the highest number found
                    if number > highest_number:
                        highest_number = number
                        latest_checkpoint = file

    return latest_checkpoint    