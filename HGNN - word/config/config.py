import yaml

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
    # Default 값 설정
    config.setdefault('training', {})
    config['training'].setdefault('batch_size', 32)
    config['training'].setdefault('lr', 0.001)
    config['training'].setdefault('weight_decay', 1e-4)
    config['training'].setdefault('epochs', 20)
    config['training'].setdefault('checkpoint_interval', 5)
    config['training'].setdefault('save_dir', 'checkpoints')
    return config
