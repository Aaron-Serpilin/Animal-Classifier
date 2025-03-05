import torch
from pathlib import Path

def save_model(
        model: torch.nn.Module, 
        target_dir: str,
        model_name: str 
    ):
    """
    Saves a PyTorch Model into its corresponding directory. 

    Args:
        model: The target PyTorch model to save
        target_dir: The directory to save the model in
        model_name: The filename for the model (the file extension should be .pt or .pth)

    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    save_model_path = target_dir_path / model_name
    torch.save(obj=model.state_dict(), f=save_model_path)
    print(f"[INFO] Model saved to: {save_model_path}")