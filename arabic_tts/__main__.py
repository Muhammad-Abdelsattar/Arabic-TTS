import os
import typer
import wandb
from omegaconf import OmegaConf
from .factory import create_trainer

cli = typer.Typer()

@cli.command()
def train(config: str = typer.Option("config.yaml", "-c", "--config", help="Configuration path."),
          continue_path: str = typer.Option("", "-p", "--continue-path", help="Path to the checkpoint to continue training."),
          wandb_key: str = typer.Option("", "-w", "--wandb-key", help="Wandb API key.")):
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
def export_model(checkpoint_path: str = typer.Option("", "-c", "--config", help="Configuration path."),):
    print("Exporting model...")
    print("The export method isn't implemented yet.")
    
if __name__ == "__main__":
    cli()
