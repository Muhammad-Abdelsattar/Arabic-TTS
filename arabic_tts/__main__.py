import os
import typer
import wandb
from omegaconf import OmegaConf
from .factory import create_trainer, create_model
from .utils import get_tokenizer_config, save_tokenizer_config

cli = typer.Typer()

@cli.command()
def train(config: str = typer.Option("config.yaml", "-c", "--config", help="Configuration path."),
          continue_path: str = typer.Option("", "--continue-path", help="Path to the checkpoint to continue training."),
          wandb_key: str = typer.Option("", "--wandb-key", help="Wandb API key.")):
    config = OmegaConf.load(config)
    if wandb_key:
        config.common.wandb_api_key = wandb_key
    if continue_path:
        config.trainer_args.continue_path = continue_path
    wandb.login(key=config.common.wandb_api_key)
    config = OmegaConf.to_container(config, resolve=True)
    trainer = create_trainer(config)
    trainer.fit()
    
    
@cli.command()
def export_model(config: str = typer.Option("config.yaml", "-c", "--config", help="Configuration path."),
                 checkpoint_path: str = typer.Option("", "--checkpoint-path", help="Path to the checkpoint to be exported."),
                 export_path: str = typer.Option("", "--export-path", help="Path to save the exported model and metadata.")):
    config = OmegaConf.load(config)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    config = OmegaConf.to_container(config, resolve=True)
    model, config = create_model(config)
    model.load_checkpoint(config=config, checkpoint_path=checkpoint_path)
    print(f"> Model loaded from {checkpoint_path}.")
    print(f"> Exporting model to {export_path}.")
    model.export_onnx(export_path)
    print(f"> Model exported to {export_path} successfully.")


if __name__ == "__main__":
    cli()
