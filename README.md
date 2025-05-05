# Image Classification with Drift Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps pipeline for image classification on Imagenette-160, featuring automated data drift detection and retraining.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         drift_detector_pipeline and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── drift_detector_pipeline   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes drift_detector_pipeline a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## 1. Team Information
-   **Team Name:** Team Zeal
-   **Team Members:**
    -   Montelongo, Esteban - EMONTEL1@depaul.edu
    -   Bandara, Sajith - SBANDARA@depaul.edu
    -   Sankar Chandrasekar, Arjun Kumar - ASANKARC@depaul.edu
-   **Course & Section:** SE 489: ML Engineering For Production (MLOps)

## 2. Project Overview
-   **Summary:** This project implements an end-to-end MLOps pipeline to train an image classification model (ResNet-18 using `timm`) on the Imagenette-160 dataset, monitor for data drift, and eventually automate retraining.
-   **Problem Statement:** Machine learning models deployed in production often suffer performance degradation over time due to changes in the underlying data distribution (data drift). This project aims to build a system that can automatically detect such drift in an image classification task and maintain model performance through automated retraining.
-   **Main Objectives (Phase 1 Focus):**
    1.  Establish a reproducible baseline training pipeline for ResNet-18 on clean Imagenette-160 data, achieving ≥85% validation accuracy.
    2.  Set up version control for data (DVC) and code (Git).
    3.  Integrate configuration management (Hydra) and experiment tracking (WandB).
    4.  Structure the codebase for future drift simulation and detection implementation.

## 3. Project Architecture Diagram
![Phase 1 Architecture Diagram](./phase1.jpg)

## 4. Phase Deliverables
-   [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
-   [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
-   [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment 

## 5. Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone git@github.com:estmon8u/team-zeal-project.git
    cd team-zeal-project
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    # Create (only once)
    python -m venv .venv
    # Activate (each time you work on the project)
    # Mac/Linux:
    source .venv/bin/activate
    # Windows Cmd:
    # .venv\Scripts\activate.bat
    # Windows PowerShell:
    # .venv\Scripts\Activate.ps1
    ```
3.  **Install Dependencies:**
    ```bash
    # Ensure pip is up-to-date
    python -m pip install --upgrade pip
    # Install project core and development dependencies
    pip install -e .[dev]
    ```
4.  **Set Up DVC Remote:**
    -   Ensure you have access to the shared Google Drive folder used for DVC storage.
    -   Authenticate DVC with Google Drive (if not already configured locally):
        ```bash
        # This might trigger browser authentication the first time
        # Or might use existing credentials if available
        dvc pull -r gdrive data/raw/imagenette2-160.tgz.dvc
        ```
        *(Note: For non-interactive environments like GitHub Actions, service account credentials configured via secrets are needed.)*
5.  **Pull Data:**
    ```bash
    # Pull the raw dataset tracked by DVC
    dvc pull
    ```

## 6. Usage Instructions

1.  **Process Raw Data:**
    -   Ensure raw data is pulled (see Setup Step 5).
    -   Run the extraction script using the Makefile:
        ```bash
        make process_data
        ```
    -   This will extract the `.tar.gz` file into `data/processed/imagenette2-160/`.

2.  **Run Baseline Model Training:**
    -   Ensure the virtual environment is activated.
    -   Ensure WandB is set up (`wandb login` might be needed once).
    -   Execute the training using the Makefile:
        ```bash
        make train
        ```
    -   Monitor progress in the terminal and on your WandB project dashboard: `https://wandb.ai/emontel1-depaul-university/zeal-imagenette-drift`
    -   Outputs (logs, best model) will be saved in the `outputs/YYYY-MM-DD/HH-MM-SS/` directory created by Hydra.

## 7. Contribution Summary
-   Esteban Montelongo: [Description of contributions for Phase 1]
-   Sajith Bandara: [Description of contributions for Phase 1]
-   Arjun Kumar Sankar Chandrasekar: [Description of contributions for Phase 1]

## 8. References & Key Tools Used
-   **Dataset:** [Imagenette-160](https://github.com/fastai/imagenette)
-   **ML Framework:** [PyTorch](https://pytorch.org/)
-   **Model Zoo:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
-   **Data Versioning:** [DVC (Data Version Control)](https://dvc.org/) + Google Drive
-   **Configuration Management:** [Hydra](https://hydra.cc/)
-   **Experiment Tracking:** [Weights & Biases (WandB)](https://wandb.ai/)
-   **Code Quality:** [Ruff](https://github.com/astral-sh/ruff)
-   **Version Control:** [Git](https://git-scm.com/) & [GitHub](https://github.com/)
-   **Orchestration (Local):** [GNU Make](https://www.gnu.org/software/make/)
-   **Python Environment:** `venv` + `pip`

---