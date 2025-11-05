# src/config.py
import os
import yaml
import logging
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    n_bayesiana: int
    umbral: float = 0.025
    ganancia: int = 780000
    estimulo: int = 20000
    n_trials: int = 35
    n_boosts: int = 1000
    n_folds: int = 5

@dataclass
class PathConfig:
    """Path configuration for different environments"""
    base_path: str
    input_data: str
    logs: str
    output: Dict[str, str]
    bayesian: Dict[str, str]
    final: Dict[str, str]
    experiments: Dict[str, str]

class ConfigurationManager:
    """Manages all configuration aspects of the project"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self._load_config()
        self.paths = self._setup_paths()
        self.model_config = self._setup_model_config()
        
    def _load_config(self) -> dict:
        """Load and validate configuration file"""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise
            
    def _setup_paths(self) -> PathConfig:
        """Set up path configuration based on environment"""
        paths_cfg = self.cfg["configuracion_paths"]
        is_gcp = self.cfg["configuracion_gcp"]["IN_GCP"]
        
        base_path = (paths_cfg["place_path"]["GCP_PATH"] 
                    if is_gcp else 
                    paths_cfg["place_path"]["LOCAL_PATH"])
                    
        return PathConfig(
            base_path=base_path,
            input_data=self._join_path(base_path, self.cfg["configuracion_competencia_1" if self.cfg["COMPETENCIA"] == 1 else "configuracion_competencia_2"]["FILE_INPUT_DATA"]),
            logs=self._join_path(base_path, paths_cfg["PATH_LOGS"]),
            output=self._get_output_paths(base_path, paths_cfg),
            bayesian=self._get_bayesian_paths(base_path, paths_cfg),
            final=self._get_final_paths(base_path, paths_cfg),
            experiments=self._get_experiment_paths(base_path, paths_cfg)
        )

    def _setup_model_config(self) -> ModelConfig:
        """Set up model configuration parameters"""
        bayes_cfg = self.cfg["configuracion_bayesiana"]
        return ModelConfig(
            n_bayesiana=bayes_cfg["N_BAYESIANA"],
            umbral=bayes_cfg.get("UMBRAL", 0.025),
            ganancia=bayes_cfg.get("GANANCIA", 780000),
            estimulo=bayes_cfg.get("ESTIMULO", 20000),
            n_trials=bayes_cfg.get("N_TRIALS", 35),
            n_boosts=bayes_cfg.get("N_BOOSTS", 1000),
            n_folds=bayes_cfg.get("N_FOLDS", 5)
        )

    @staticmethod
    def _join_path(base: str, path: str) -> str:
        """Join base path with subpath"""
        return str(Path(base) / path)

    def _get_output_paths(self, base_path: str, paths_cfg: dict) -> Dict[str, str]:
        """Get output directory paths"""
        output_cfg = paths_cfg["path_outputs"]
        return {k: self._join_path(base_path, v) for k, v in output_cfg.items()}

    def _get_bayesian_paths(self, base_path: str, paths_cfg: dict) -> Dict[str, str]:
        """Get Bayesian optimization output paths"""
        bayes_cfg = paths_cfg["path_outputs_bayesian"]
        return {k: self._join_path(base_path, v) for k, v in bayes_cfg.items()}

    def _get_final_paths(self, base_path: str, paths_cfg: dict) -> Dict[str, str]:
        """Get final output paths"""
        final_cfg = paths_cfg["path_outputs_finales"]
        return {k: self._join_path(base_path, v) for k, v in final_cfg.items()}

    def _get_experiment_paths(self, base_path: str, paths_cfg: dict) -> Dict[str, str]:
        """Get experiment output paths"""
        exp_cfg = paths_cfg["path_outputs_experimentos"]
        return {k: self._join_path(base_path, v) for k, v in exp_cfg.items()}

    @property
    def experiment_number(self) -> str:
        """Get experiment number based on process type"""
        return (self.cfg["configuracion_bayesiana"]["N_BAYESIANA"] 
                if self.cfg["PROCESO_PPAL"] == "bayesiana"
                else self.cfg["configuracion_experimentos"]["N_EXP"])

    @property
    def training_months(self) -> list:
        """Get training months configuration"""
        comp_cfg = (self.cfg["configuracion_competencia_1"] 
                   if self.cfg["COMPETENCIA"] == 1
                   else self.cfg["configuracion_competencia_2"])
        return comp_cfg.get("MES_TRAIN", [202101, 202102, 202103])

# Create global configuration instance
try:
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    config = ConfigurationManager(CONFIG_PATH)
except Exception as e:
    logger.error(f"Failed to initialize configuration: {e}")
    raise

# Export commonly used values
GANANCIA = config.model_config.ganancia
ESTIMULO = config.model_config.estimulo
PATHS = config.paths