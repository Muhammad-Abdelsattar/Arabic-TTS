import os
import json

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