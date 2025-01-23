import os
from pathlib import Path
import json
import typer
import wandb
from omegaconf import OmegaConf
from .factory import create_trainer, create_model, create_main_config
from .utils import get_tokenizer_config, save_tokenizer_config, get_latest_checkpoint

cli = typer.Typer()

@cli.command()
def train(config: str = typer.Option("config.yaml", "-c", "--config", help="Configuration path."),
          continue_path: str = typer.Option("", "--continue-path", help="Path to the checkpoint to continue training."),
          restore_path: str = typer.Option("", "--restore-path", help="Path to the checkpoint to start fine-tuning from."),
          wandb_key: str = typer.Option("", "--wandb-key", help="Wandb API key.")):
    """
    Trains or Fine-tunes a TTS model.
    """


    config = OmegaConf.load(config)
    used_path = ""

    if wandb_key:
        config.common.wandb_api_key = wandb_key
    if continue_path:
        config.trainer_args.continue_path = continue_path
        used_path = continue_path
    if restore_path:
        path = os.path.join(restore_path, get_latest_checkpoint(restore_path))
        config.trainer_args.restore_path = path
        used_path = restore_path

    if(continue_path and restore_path):
        print(f"\t|> ! Both restore_path and continue_path are provided, restore_path takes precedence.")
        config.trainer_args.continue_path = ""
        used_path = restore_path

    config = OmegaConf.to_container(config, resolve=True)
    main_config = create_model(config)[1].to_dict()

    if used_path:
        if not os.path.exists(os.path.join(used_path, "config.json")):
            print(f"\t|> ! No config.json found in {used_path}, using provided config.")
            with open(os.path.join(used_path, "config.json"), "w") as f:
                json.dump(main_config, f)
        else:
            with open(os.path.join(used_path, "config.json"), "r") as f:
                ckpt_config = json.load(f)
                
            #update ckpt_config with the provided config
            ckpt_config.update(main_config)
            
            with open(os.path.join(used_path, "config.json"), "w") as f:
                json.dump(ckpt_config, f)

    wandb.login(key=config["common"]["wandb_api_key"])
    trainer = create_trainer(config)
    trainer.fit()
    
    
@cli.command()
def export_model(config: str = typer.Option("config.yaml","--config-path", help="The path to the config file used to train the checkpoint model. That must explicitly be the one used to train the model."),
                 checkpoint_path: str = typer.Option("", "--checkpoint-path", help="Path to the checkpoint to be exported."),
                 export_path: str = typer.Option("", "--export-path", help="Path to save the exported model and metadata.")):

    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config, resolve=True)
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")

    #The checkpoint config will be used to extract the model architecture.
    # with open(path.parent.absolute()/"config.json", "r") as f:
    #     temp_config = json.load(f)
        

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    #Ensure that the created model architecture is the same as the one in the checkpoint
    # config["model_args"] = temp_config["model_args"]

    model, config = create_model(config)
    model.load_checkpoint(config=config, checkpoint_path=checkpoint_path)
    print(f"> Model loaded from {checkpoint_path}.")
    print(f"> Exporting model to {export_path}.")
    model.export_onnx(os.path.join(export_path, "model.onnx"))
    print(f"> Exporting tokenizer config to {export_path}.")
    save_tokenizer_config(os.path.join(export_path, "tokenizer_config.json"), get_tokenizer_config(model))
    print(f"> Model exported to {export_path} successfully.")

if __name__ == "__main__":
    cli()
