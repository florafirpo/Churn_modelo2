# AI Coding Assistant Instructions for Churn Prediction Bank Project

## Project Overview
This is a machine learning project focused on bank customer churn prediction using LightGBM and XGBoost models. The project implements multiple experimental approaches and uses Bayesian optimization for hyperparameter tuning.

## Key Architecture Components

### Configuration Management
- All configuration is managed through `config.yaml` at the project root
- Environment-specific paths (GCP vs local) are handled in config
- Configuration is loaded and parsed in `src/config.py`

### Core Components
1. Data Pipeline (`src/`)
   - `loader.py`: Data loading and initial preprocessing
   - `preprocesamiento.py`: Data preprocessing including binary conversion and train/test splitting
   - `feature_engineering.py`: Feature creation and transformations
   - `constr_lista_cols.py`: Column list management

2. Model Training (`src/`)
   - `lgbm_train_test.py`: LightGBM model training and evaluation
   - `xgb_train_test.py`: XGBoost model training and evaluation
   - `lgbm_optimizacion.py`, `xgb_optimizacion.py`: Model optimization

3. Experimentation (`src_experimentos/`, `src_bayesianas/`)
   - Multiple experiment implementations with different approaches
   - Bayesian optimization experiments for model tuning

## Development Workflows

### Project Setup
1. Install dependencies:
```bash
pip install -r requirements.txt  # For local development
pip install -r vm_requirements.txt  # For VM/cloud deployment
```

2. Configure environment:
- Set `IN_GCP: False` in `config.yaml` for local development
- Update paths in `config.yaml` as needed

### Running Experiments
1. Update experiment configuration in `config.yaml`:
   - Set `COMPETENCIA` (1 or 2)
   - Set `PROCESO_PPAL` ("bayesiana" | "experimento" | "prediccion_final" | "test" | "analisis_exploratorio")
   - Configure experiment number in `configuracion_experimentos.N_EXP`

2. Execute main script:
```bash
python main.py
```

## Key Conventions

### Feature Engineering Pattern
1. Load data via `cargar_datos()`
2. Apply feature transformations in sequence:
   ```python
   df = feature_engineering_lag(df, cols, lag_periods)
   df = feature_engineering_delta(df, cols, periods)
   df = feature_engineering_ratio(df, ratio_cols)
   ```

### Model Training Pattern
1. Convert data to binary format using `conversion_binario()`
2. Split data using `split_train_test_apred()`
3. Load/optimize hyperparameters
4. Train and evaluate models

### Logging
- Use the configured logger with appropriate levels
- Global logs in `global_log/registro_global.txt`
- Experiment-specific logs in `logs/` directory

## Integration Points
- GCP for cloud deployment (controlled via `IN_GCP` flag)
- Data input from CSV files (local) or cloud storage (GCP)
- Output artifacts stored in configured paths

Always check `config.yaml` for current settings when making changes.