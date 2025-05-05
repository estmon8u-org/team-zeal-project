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
    1.  Establish a reproducible baseline training pipeline for ResNet-18 on clean Imagenette-160 data, achieving ≥85% validation accuracy. *(Achieved: 96.76%)*
    2.  Set up version control for data (DVC) and code (Git). *(Completed)*
    3.  Integrate configuration management (Hydra) and experiment tracking (WandB). *(Completed)*
    4.  Structure the codebase for future drift simulation and detection implementation. *(Completed)*
    5.  Implement basic unit tests for data handling. *(Completed)*

## 3. Project Architecture Diagram (Phase 1)
![Phase 1 Architecture Diagram](./phase1.jpg)
*(This diagram represents the components set up in Phase 1: Data acquisition/versioning with DVC/G-Drive, Code versioning with Git/GitHub, Training pipeline using PyTorch/timm, Configuration via Hydra, Experiment Tracking via WandB, Unit Testing via Pytest, and local execution via Make.)*

## 4. Phase Deliverables
-   [X] [PHASE1.md](./PHASE1.md): Project Design & Model Development *(Completed)*
-   [ ] PHASE2.md: Enhancing ML Operations *(Upcoming)*
-   [ ] PHASE3.md: Continuous ML & Deployment *(Upcoming)*

## 5. Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/estmon8u/team-zeal-project.git # Or use SSH URL
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
    
    # Option 1: Using pip directly
    pip install -e . # Use pyproject.toml for install
    
    # Option 2: Using make command
    make requirements # This also uses pip install -e . behind the scenes
    
    # Optional: Install dev dependencies if needed separately (ruff, pytest etc. included above via pyproject)
    # pip install -r requirements.txt # Can also use this, but pyproject.toml is preferred
    ```
4.  **Accept WandB Team Invitation & Login (First Time Only):**
    *   Ensure you have accepted the invitation to join the **`emontel1-depaul-university`** team/entity on WandB (check your email or WandB notifications). You'll need a free WandB account associated with the invited email address.
    *   Run `wandb login` in your terminal and follow the prompts to authenticate your WandB account if you haven't used WandB on this machine before. This ensures your runs log to the shared team project.

5.  **Set Up DVC Remote (Google Drive - First Time Only):**
    *   **Permissions:** Ensure the Google Drive folder (ID: `19qyjvhry7pP9AF4q03hbKl4M5EWhrtk2`, specified in `.dvc/config`) has been shared with your `@depaul.edu` / `@gmail.com` Google account with **Editor** permissions by the project lead.
    *   **Authenticate DVC:** The first time you run a command like `dvc pull`, you will be prompted to authenticate DVC with Google Drive via your browser. Follow the instructions, making sure to log in with your appropriate Google account and grant permissions.
        ```bash
        # Example command to trigger authentication if needed:
        dvc pull data/raw/imagenette2-160.tgz.dvc
        ```
6.  **Pull DVC Tracked Files (Raw Data & Baseline Model):**
    ```bash
    # Pull all files tracked by DVC using the authenticated remote
    dvc pull
    # Or pull specific files:
    # dvc pull data/raw/imagenette2-160.tgz.dvc
    # dvc pull models/resnet18_baseline_v1.pth.dvc
    ```

## 6. Usage Instructions

1.  **Activate Environment:**
    ```bash
    source .venv/bin/activate # Or Windows equivalent
    ```
2.  **Process Raw Data (Extract Archive):**
    -   Ensure raw data is pulled (Setup Step 6).
    -   Run the extraction using the Makefile:
        ```bash
        make process_data
        ```
    -   This extracts `data/raw/imagenette2-160.tgz` into `data/processed/imagenette2-160/`.
3.  **Run Baseline Model Training:**
    -   Ensure Weights & Biases is set up (run `wandb login` once if needed).
    -   Execute the training using the Makefile (uses `conf/config.yaml` by default):
        ```bash
        make train
        ```
    -   Monitor progress in the terminal and on your WandB project dashboard: [https://wandb.ai/emontel1-depaul-university/zeal-imagenette-drift](https://wandb.ai/emontel1-depaul-university/zeal-imagenette-drift)
    -   Hydra will create an output directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`) containing logs (`train.log`), Hydra configs (`.hydra/`), and the saved `best_model.pth`. The best model from the run is automatically versioned by DVC after manual addition (see step 5.2 in PHASE1.md).
    -   **Override Config:** You can change parameters via the command line using Hydra syntax, e.g.:
        ```bash
        python -m drift_detector_pipeline.modeling.train training.epochs=5 training.learning_rate=0.0005
        ```
4.  **Run Linting/Formatting:**
    ```bash
    make lint  # Check formatting and linting using Ruff
    make format # Apply formatting and fix linting issues using Ruff
    ```
5.  **Run Tests:**
    ```bash
    make test # Executes unit tests defined in the tests/ directory using pytest
    ```

## 7. Contribution Summary (Phase 1)
-   **Esteban Montelongo:** DVC setup & data versioning, `dataset.py` (extraction, transforms, dataloaders), initial documentation structure (`README.md`, `PHASE1.md`), architecture diagram, model DVC tracking, unit test implementation.
-   **Sajith Bandara:** Hydra integration (`conf/config.yaml`, `train.py` decorator/config usage), `train.py` core structure (model loading, optimizer, scheduler, loop), Makefile setup (`train`, `process_data` rules), model saving path correction.
-   **Arjun Kumar Sankar Chandrasekar:** WandB integration (`wandb.init`, `wandb.log`), dependency management (`pyproject.toml`, `requirements.txt`), `ruff` configuration and code formatting, testing infrastructure setup and test contributions.


## 8. References & Key Tools Used
-   **Dataset:** [Imagenette-160 (v2)](https://github.com/fastai/imagenette)
-   **ML Framework:** [PyTorch](https://pytorch.org/)
-   **Model Zoo:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
-   **Data Versioning:** [DVC (Data Version Control)](https://dvc.org/) + Google Drive
-   **Configuration Management:** [Hydra](https://hydra.cc/)
-   **Experiment Tracking:** [Weights & Biases (WandB)](https://wandb.ai/)
-   **Code Quality:** [Ruff](https://github.com/astral-sh/ruff) (Linting & Formatting), [Pytest](https://pytest.org/) (Testing)
-   **Version Control:** [Git](https://git-scm.com/) & [GitHub](https://github.com/)
-   **Build/Task Runner:** [GNU Make](https://www.gnu.org/software/make/)
-   **Python Environment:** `venv` + `pip`
-   **Project Template:** [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)