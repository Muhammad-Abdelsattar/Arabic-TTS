import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def create_model_args(config:dict):
    return VitsArgs(**config)

def create_audio_config(config:dict):
    return VitsAudioConfig(**config)

def create_characters_config(config:dict):
    return CharactersConfig(**config)

def create_dataset_config(config:dict):
    return BaseDatasetConfig(**config)

def create_tokenizer(config:dict):
    return TTSTokenizer(**config)

def create_trainer_args(config:dict):
    return TrainerArgs(**config)

def create_main_config(config:dict):
    model_args = create_model_args(config.get("model_args", {}))
    audio_config = create_audio_config(config.get("audio_config", {}))
    characters_config = create_characters_config(config.get("characters_config", {}))
    main_config = config.get("main", {})
    return VitsConfig(model_args=model_args,
                      audio=audio_config,
                      characters=characters_config,
                      **main_config)

def create_model(config:dict):
    main_config = create_main_config(config)
    
    audio_processor = AudioProcessor.init_from_config(main_config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, main_config = TTSTokenizer.init_from_config(main_config)

    # init model
    model = Vits(main_config, audio_processor, tokenizer, speaker_manager=None)

    return model, main_config

def create_trainer(config:dict):
    model, main_config = create_model(config)
    dataset_config = create_dataset_config(config.get("dataset_config", {}))
    train_samples, eval_samples = load_tts_samples(dataset_config,
                                                   eval_split=True,
                                                   eval_split_max_size=main_config.eval_split_max_size,
                                                   eval_split_size=main_config.eval_split_size,
                                                   )
    trainer_args = create_trainer_args(config.get("trainer_args", {}))

    trainer = Trainer(trainer_args,
                      main_config,
                      output_path=config.get("output_path", ""),
                      model=model,
                      train_samples=train_samples,
                      eval_samples=eval_samples)
    
    return trainer


