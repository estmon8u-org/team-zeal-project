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
    
    # Install project dependencies (includes DVC, PyTorch, etc.)
    pip install -e . 
    ```
4.  **Accept WandB Team Invitation & Login (First Time Only - For Original Team):**
    *   If you are part of the original project team and want to log to the shared WandB space: Ensure you have accepted the invitation to join the **`emontel1-depaul-university`** team/entity on WandB.
    *   Run `wandb login` in your terminal and follow the prompts to authenticate.
    *   *For general users setting up their own instance, you will configure your own WandB project/entity later if desired (see `conf/config.yaml`).*

5.  **Setting Up DVC (Data Version Control) with Your Own Remote Storage:**

    This project uses DVC to manage large data files and models. The Git repository contains `.dvc` metafiles that point to the actual data. If you want to replicate the project with your own data storage, follow these steps (using Google Drive as an example):

*   **5.1. Initialize DVC (if starting a project from absolute scratch):**
    Your cloned repository should already have a `.dvc` directory and configuration. If it didn't (e.g., you were starting a new project based on this one), you would run:
    ```bash
    dvc init
    ```
    This creates the `.dvc` directory and a basic configuration.

*   **5.2. Configure Your DVC Remote (Example: Google Drive):**
    1.  **Create a folder in your Google Drive** where you want to store the DVC-tracked files.
    2.  **Get the Folder ID:** Open the folder in Google Drive. The ID is the last part of the URL (e.g., if the URL is `https://drive.google.com/drive/folders/ABCDEFG12345`, the ID is `ABCDEFG12345`).
    3.  **Modify or Add DVC Remote Configuration:**
        The existing `.dvc/config` file in this repository points to the original authors' Google Drive. You'll need to update it to point to *your* Google Drive folder, or add a new remote.

        **Option A: Modify the existing 'gdrive' remote (Recommended for simplicity if you're the primary user of your fork):**
        ```bash
        dvc remote modify gdrive url gdrive://YOUR_GOOGLE_DRIVE_FOLDER_ID
        ```
        Replace `YOUR_GOOGLE_DRIVE_FOLDER_ID` with the ID you obtained.

        **Option B: Add a new remote (if you want to keep the original remote config for reference):**
        ```bash
        dvc remote add mygdrive gdrive://YOUR_GOOGLE_DRIVE_FOLDER_ID
        dvc remote default mygdrive # Set your new remote as the default
        ```

    4.  **Set DVC Google Drive Credentials (Optional but Recommended for Automation/CI):**
        By default, DVC will use `gdrive_use_default_credential true` which prompts for browser authentication. For more control or non-interactive environments, you might configure specific OAuth credentials.
        ```bash
        # Example: Use default browser authentication (usually sufficient for personal use)
        dvc remote modify gdrive gdrive_use_default_credential true 
        # If you added 'mygdrive', use:
        # dvc remote modify mygdrive gdrive_use_default_credential true
        ```
        For advanced credential setup (e.g., service accounts or your own client ID/secret), refer to the [official DVC Google Drive documentation](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive).
        *Security Note: Be careful with client secrets. Do not commit them directly to Git. Use `.dvc/config.local` (which is in `.dvc/.gitignore`) for sensitive remote configurations if needed.*

    5.  **Authenticate DVC with Google Drive:**
        The first time you run a DVC command that interacts with the remote (like `dvc push` or `dvc pull` in the next steps), you will likely be prompted to authenticate via your browser. Follow the instructions, making sure to log in with your Google account that has access to the folder you created.

*   **5.3. Download and Version the Raw Dataset:**
    1.  **Download the Dataset:** The Imagenette-160 (v2 split) dataset can be downloaded from the [FastAI GitHub repository](https://github.com/fastai/imagenette).
        Direct link to `imagenette2-160.tgz`: [https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz)
    2.  **Place the Dataset:** Create the directory `data/raw/` if it doesn't exist, and place the downloaded `imagenette2-160.tgz` file into it.
        The path should be: `data/raw/imagenette2-160.tgz`
    3.  **Track with DVC:** Tell DVC to track this file. This will create/overwrite `data/raw/imagenette2-160.tgz.dvc`.
        ```bash
        dvc add data/raw/imagenette2-160.tgz
        ```
    4.  **Commit the Metafile to Git:**
        ```bash
        git add data/raw/imagenette2-160.tgz.dvc .dvc/config 
        git commit -m "feat: Track raw Imagenette dataset with DVC and update remote config"
        ```
    5.  **Push to Your DVC Remote:**
        ```bash
        dvc push
        ```
        This uploads the `imagenette2-160.tgz` (as managed by DVC) to your configured Google Drive folder.

*   **5.4. Baseline Model (You will train your own):**
    The `.dvc` file for the baseline model (`models/resnet18_baseline_v1.pth.dvc`) in this repository points to the original authors' DVC storage.
    **You will train your own baseline model in the "Usage Instructions" (Step 3: Run Baseline Model Training).**
    After training, your best model will be saved (e.g., in `outputs/YYYY-MM-DD/HH-MM-SS/best_model.pth`). You can then optionally track this model with DVC:
    ```bash
    # Example after training:
    # dvc add outputs/YYYY-MM-DD/HH-MM-SS/best_model.pth -o models/my_resnet18_baseline_v1.pth
    # git add models/my_resnet18_baseline_v1.pth.dvc
    # git commit -m "model: Add my trained baseline model to DVC"
    # dvc push
    ```
    For now, you don't need to pull any pre-existing DVC-tracked model. If you see a DVC file for a model, you can generally ignore it until you've trained your own.


## 6. Usage Instructions

1.  **Activate Environment:**
    ```bash
    source .venv/bin/activate # Or Windows equivalent
    ```
2.  **Process Raw Data (Extract Archive):**
    -   Ensure the raw dataset `data/raw/imagenette2-160.tgz` is present (either by following Step 5.3 above or, if you have access to a DVC remote where it's already pushed, by running `dvc pull data/raw/imagenette2-160.tgz.dvc`).
    -   Run the extraction using the Makefile:
        ```bash
        make process_data
        ```
    -   This extracts `data/raw/imagenette2-160.tgz` into `data/processed/imagenette2-160/`.

3.  **Run Baseline Model Training:**
    *(Ensure WandB is set up if you want to log to your own WandB account - see `conf/config.yaml` to set your project/entity, and run `wandb login` once if needed).*
    -   Execute the training using the Makefile (uses `conf/config.yaml` by default):
        ```bash
        make train
        ```
    -   Monitor progress in the terminal and on your WandB project dashboard (if configured).
    -   Hydra will create an output directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`) containing logs, Hydra configs, and the saved `best_model.pth`.

    *   **Override Config (Example):**
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