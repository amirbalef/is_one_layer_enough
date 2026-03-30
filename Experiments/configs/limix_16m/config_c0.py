from .config_base import config_base

def generate_config_list():
    config = config_base.copy()
    config["model_name"] = f"{config['model']}_c0"
    config["desc"] = f"{config['model']}_default"
    config["embedding_distances"] = True
    return [config]