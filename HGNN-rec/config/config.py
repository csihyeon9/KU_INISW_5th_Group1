# config/config.py
import os
import yaml
import os.path as osp
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for the project."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        # YAML에 custom tag handler 추가
        def join_path(loader, node):
            seq = loader.construct_sequence(node)
            return str(Path(*seq))
        
        # Custom tag 등록
        yaml.add_constructor('!join', join_path, Loader=yaml.SafeLoader)
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._check_required_paths(config)
        return config

    def _validate_config(self):
        """Validate configuration values and their dependencies."""
        # 모델 설정 검증
        model_config = self.config.get('model', {})
        assert all(k in model_config for k in ['in_dim', 'hidden_dim', 'embedding_dim']), \
            "Missing required model configuration"

        # Hypergraph 설정 검증
        hg_config = self.config.get('hypergraph', {})
        assert hg_config.get('construction_method') in ['natural', 'knn'], \
            "Invalid hypergraph construction method"

        # 확장 기능 의존성 검증
        extensions = self.config.get('extensions', {})
        if extensions.get('use_kf_deberta'):
            assert model_config['in_dim'] == 768, \
                "Input dimension must be 768 when using KF-DeBERTa"

    def _check_required_paths(self, config: Dict[str, Any]):
        """Check if required data paths exist."""
        data_root = config.get('data_root')
        if data_root:
            os.makedirs(data_root, exist_ok=True)  # 디렉토리가 없으면 생성

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return self.config.get('training', {})

    def get_hypergraph_config(self) -> Dict[str, Any]:
        """Get hypergraph-specific configuration."""
        return self.config.get('hypergraph', {})

    def get_recommendation_config(self) -> Dict[str, Any]:
        """Get recommendation-specific configuration."""
        return self.config.get('recommendation', {})