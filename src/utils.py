from src.DiffusionModel import DiffusionModel


def get_model_class(name: str):
    if name == "diffusion":
        return DiffusionModel
    else:
        raise ValueError(f"Unknown model name {name=}")
