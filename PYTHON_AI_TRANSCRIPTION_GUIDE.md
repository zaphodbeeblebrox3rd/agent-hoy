# Python AI and Data Science Transcription Improvements

Transcription corrections and phrase hints for Python AI, data science, and programming terms.

## What is included

### Transcription corrections

- File: `conf/corrections.conf.example` (override with `conf/corrections.conf`)
- Integration: loaded at startup via `config_loader.load_transcription_corrections()`
- Coverage: Python libraries, AI/ML frameworks, programming and cloud terms
- Size: 700+ mappings across Python, HPC, and hybrid-cloud categories

### Library categories

#### Core data science

- pandas (panda, pan does, pen does)
- numpy (num pie, num pi, numb pie)
- scipy (sigh pie, sci pi, psy pie)
- matplotlib (mat plot lib, matter plot lib)
- scikit-learn (psychic kit learn, psy kit learn)
- seaborn (sea born, see born)
- plotly (plot lee)

#### Machine learning frameworks

- tensorflow (tensor flow)
- pytorch (py torch)
- keras (care ass)
- xgboost (extra boost, X G boost)
- lightgbm (light GBM)
- catboost (cat boost)

#### Deep learning and MLOps

- hugging face (hub space)
- transformers
- mlflow (ML flow)
- wandb (weights and biases, one d be)
- tensorboard (tensor board)

#### Computer vision and NLP

- opencv (open CV)
- pillow (pill ow)
- spacy (spacey, space E)
- nltk, gensim (gen sim), textblob (text blob)

#### Data processing and visualization

- polars (pole airs), pyarrow (pie arrow), dask (day task)
- streamlit (stream lit), gradio (grey deo), bokeh (boca)

### Python programming terms

- Core: python (pie thon), pip (pie P), conda (condo), jupyter (jupiter)
- Environments: virtualenv, venv, anaconda
- OOP and syntax: class, method, function, def, __init__, elif, lambda, yield
- Structures: dataframe, dictionary, tuple, list

### Cloud and infrastructure

- AWS: aws, ec2, s3, lambda, sagemaker
- Docker / Kubernetes: docker, dockerfile, docker-compose, kubernetes (k8s, K eight s)
- Git: git, github, gitlab

## How it works

1. Corrections load from `conf/corrections.conf` if present, else the shipped `.example` file.
2. Each STT result is passed through `correct_transcription_errors` in `main.py`.
3. With `VERBOSE=true`, original and corrected text are logged.

## Recognition phrases

`transcription_config.py` includes technical phrases for the Python ecosystem, infrastructure, data formats, and databases to improve STT hints.

## Custom corrections

Copy the example and edit the override (gitignored):

```bash
cp conf/corrections.conf.example conf/corrections.conf
```

```conf
# Your custom corrections
my weird term = correct term
another mistake = proper spelling
```

Rollback: delete `conf/corrections.conf` to return to the shipped example defaults.

## File locations

- Corrections: `conf/corrections.conf.example` / `conf/corrections.conf`
- Technical phrases: `transcription_config.py`
- Application logic: `main.py` (`correct_transcription_errors`)
- Loader: `config_loader.py`

## Related docs

- [HPC_HYBRID_CLOUD_ADDITIONS.md](HPC_HYBRID_CLOUD_ADDITIONS.md) for HPC and cloud terms
- [TRANSCRIPTION_UPGRADE_PLAN.md](TRANSCRIPTION_UPGRADE_PLAN.md) for STT architecture status
