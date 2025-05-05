# PHASE 1: Project Design & Model Development

## 1. Project Proposal

### **1.1 Project Scope and Objectives**

*   **Problem Statement:** Develop an MLOps pipeline capable of training an image classifier, detecting performance degradation due to common data corruptions (drift), and automatically triggering retraining to maintain accuracy.
*   **Project Objectives:**
    1.  Achieve a baseline classification accuracy of ≥ 85% on the clean Imagenette-160 validation split using a fine-tuned ResNet-18 model.
    2.  Implement a monitoring system capable of reliably detecting simulated drift scenarios (e.g., noise, brightness shifts) based on statistical tests or pre-defined thresholds. *(Note: Drift detection implementation is planned for Phase 2).*
    3.  Establish an automated retraining pipeline triggered by the detection of significant data drift. *(Note: Automation implementation is planned for Phase 2/3).*
*   **Success Metrics:**
    *   Model Performance: Top-1 Classification Accuracy (Baseline achieved in Phase 1).
    *   Drift Detection: Drift Detection Rate / F1-score on simulated drift data (Evaluated in Phase 2).
    *   Pipeline Automation: Successful execution rate of the automated retraining workflow (Evaluated in Phase 2/3).

### **1.2 Selection of Data**

*   **Dataset(s) chosen and justification:**
    *   Selected **Imagenette-160 (v2 split)**. Contains ~13k train/~5k val images across 10 ImageNet classes.
    *   **Justification:** Manageable size (~102MB download size), realistic 160px resolution suitable for pre-trained models (`timm`) and meaningful drift simulation compared to lower-res alternatives like CIFAR-10.
*   **Data source(s) and access method:**
    *   Downloaded from the [FastAI GitHub repository](https://github.com/fastai/imagenette) (`.tar.gz` archive).
    *   Raw data (`imagenette2-160.tgz`) versioned using DVC with Google Drive as remote storage. Metafile: `data/raw/imagenette2-160.tgz.dvc`.
    *   Accessed in code via `torchvision.datasets.ImageFolder` after extraction by `drift_detector_pipeline/dataset.py` (run via `make process_data`).
*   **Preprocessing steps:**
    *   **Implemented:** Extraction of `.tar.gz` archive via `dataset.py`. Standard image transformations (Resize(256), CenterCrop(224) for val / RandomResizedCrop(224) + RandomHorizontalFlip for train, `ToTensor`, ImageNet normalization) applied via `torchvision.transforms` within the `DataLoader` setup (`get_transforms` function in `dataset.py`).
    *   **Planned (Phase 2):** Integration of drift-specific transformations (e.g., brightness, noise) configurable via Hydra.

### **1.3 Model Considerations**

*   **Chosen Model Architecture:** **ResNet-18**
*   **Rationale for Model Choice:** Strong baseline performance, computational efficiency suitable for course hardware/Colab, excellent pre-trained weights available via `timm`.
*   **Architectural Summary:** Standard ResNet-18 architecture with residual blocks ([He et al., 2015](https://arxiv.org/abs/1512.03385)).
*   **Source/Implementation:** Utilized `timm.create_model('resnet18', pretrained=True)` ([timm GitHub Repo](https://github.com/huggingface/pytorch-image-models)) via `train.py`, replacing the final classifier head for 10 classes (specified in `conf/config.yaml`).
*   **Implementation Plan (Phase 1):** Loaded pre-trained model, adapted classifier, implemented basic fine-tuning loop (optimizer: AdamW, scheduler: CosineAnnealingLR configured via Hydra), integrated experiment tracking (WandB). *Status: Core logic implemented and functional in `drift_detector_pipeline/modeling/train.py`.*

### **1.4 Open-source Tools**

*   **Third-Party Package Selected:** `timm` (PyTorch Image Models)
*   **Brief description of how/why used:** Selected `timm` ([timm GitHub Repo](https://github.com/huggingface/pytorch-image-models)) for its easy API (`timm.create_model`) to load high-quality pre-trained ResNet-18 weights for transfer learning on Imagenette-160, as implemented in `train.py`.

## 2. Code Organization & Setup

### **2.1 Repository Setup**

*   **GitHub Repository:** Central repository created at: [https://github.com/estmon8u/team-zeal-project](https://github.com/estmon8u/team-zeal-project)
*   **Project Structure:** Standard structure initialized using Cookiecutter Data Science template, including directories like `data/`, `models/`, `drift_detector_pipeline/` (source code), `conf/`, `tests/`, `docs/` and key files (`Makefile`, `.gitignore`, `pyproject.toml`).

### **2.2 Environment Setup**

*   **Python Virtual Environment:** Managed using `venv` and `pip`. Activated via `source .venv/bin/activate` (Mac/Linux) or equivalent (Windows).
*   **Dependency Management:** Dependencies defined in `pyproject.toml` (for core structure and `flit`) and installed via `pip install -e .[dev]`. A full list including transitive dependencies is in `requirements.txt`. Key MLOps tools (`dvc[gdrive]`, `hydra-core`, `wandb`, `ruff`, `timm`) are included.
*   **Development Platform:** Google Colab used for GPU-accelerated training runs; local environment used for code development and testing.

## 3. Version Control & Collaboration

### **3.1 Git Usage**

*   **Repository:** All code, configuration (`conf/config.yaml`), documentation (`README.md`, `PHASE1.md`), and DVC meta-files (`.dvc/config`, `data/raw/imagenette2-160.tgz.dvc`) tracked in the GitHub repository linked above.
*   **Commit Practices:** Followed standard practices for frequent, atomic commits with descriptive messages (see `.git/logs/HEAD`).
*   **Branching Strategy:** Development occurred primarily on `main` for Phase 1 setup. Branching (e.g., `feature/drift-simulation`, `feature/monitoring`) planned for future features/experiments.
*   **Merging:** Pull Requests to be used for feature integration in later phases.

### **3.2 Team Collaboration**

*   **Roles (Phase 1):**
    *   Esteban Montelongo: DVC setup, Data extraction/loading script (`dataset.py`), Initial `README.md` / `PHASE1.md` structure.
    *   Sajith Bandara: Hydra setup (`config.yaml`), Training script structure (`train.py`), Makefile setup & `process_data` rule.
    *   Arjun Kumar Sankar Chandrasekar: WandB integration in `train.py`, `pyproject.toml`/`requirements.txt` management, `ruff` setup and formatting.
*   **Communication:** Primary communication via MS Teams channel 'Team Zeal SE489'. *(Adjust if different)*
*   **Code Reviews:** Conducted informally for Phase 1 setup. Formal Pull Request reviews planned for Phase 2 onwards.
*   **Merge Conflict Resolution:** N/A for Phase 1. Standard Git merge/rebase strategy planned.

## 4. Data Handling

### **4.1 Data Preparation**

*   **Data Acquisition & Versioning:** Imagenette-160 (`.tar.gz`) downloaded to `data/raw/`, tracked by DVC (metafile `data/raw/imagenette2-160.tgz.dvc`), and pushed to the configured Google Drive remote. Raw data directory (`/data/raw/*`) is correctly ignored by Git.
*   **Preprocessing Script:** Raw data extraction logic implemented in `drift_detector_pipeline/dataset.py` (`extract_data` function), runnable via `make process_data`. Extracted data resides in `data/processed/imagenette2-160/` (ignored by Git).
*   **Dataset Loading:** PyTorch `DataLoaders` implemented in `drift_detector_pipeline/dataset.py` (`get_dataloaders` function) using `torchvision.datasets.ImageFolder` on the processed data path, applying transformations defined in `get_transforms`. Parameters (batch size, workers) configured via Hydra (`conf/config.yaml`).

### **4.2 Data Documentation**

*   **README:** Basic structure included in `README.md` (Setup/Usage sections).
*   **Script Documentation:** *[Status: In Progress]* Docstrings and comments added to `dataset.py` and `train.py`, but require final review and completion for full clarity.

## 5. Model Training

### **5.1 Training Infrastructure**

*   **Platform:** Baseline training executed successfully on Google Colab GPU. *(Adjust if different)*
*   **Environment:** Setup reproducible via cloning repo and `pip install -e .[dev]`.
*   **Configuration:** Managed centrally via Hydra (`conf/config.yaml`).
*   **Experiment Tracking:** Weights & Biases (WandB) used for logging metrics (loss, accuracy), hyperparameters, and system stats. Runs logged to project `zeal-imagenette-drift` under entity `emontel1-depaul-university`.

### **5.2 Initial Model Training & Evaluation**

*   **Training Script:** Implemented in `drift_detector_pipeline/modeling/train.py`, integrating Hydra, WandB, `timm`, and data loaders. Runnable via `make train`.
*   **Baseline Training:** *[Status: Completed]* Initial training run executed for 10 epochs. Baseline validation accuracy goal (≥85%) achieved/not achieved [**Team Zeal: Please update with your actual result here, e.g., "Achieved 87.5% validation accuracy..."**]. Link to relevant WandB run: [**Team Zeal: Add link to your WandB run here**].
*   **Evaluation:** Top-1 Accuracy on validation set tracked per epoch via WandB.
*   **Model Saving:** Best model checkpoint (based on validation accuracy) saved automatically by `train.py` to the Hydra output directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/best_model.pth`). *(Note: DVC tracking for the saved model artifact should be added in subsequent steps/phases).*

## 6. Documentation & Reporting

### **6.1 Project README**

*   **Status:** *[Status: Updated]* Main `README.md` updated with setup, usage, architecture diagram, and tool list based on Phase 1 progress. Contribution summary needs final input from team members.

### **6.2 Code Documentation**

*   **Docstrings:** *[Status: In Progress]* Present in key functions (`train.py`, `dataset.py`), but need review for completeness and consistency.
*   **Inline Comments:** *[Status: In Progress]* Added for some complex parts, review needed.
*   **Code Styling & Formatting (`ruff`):** *[Status: Completed]* Code formatted using `ruff format` (commit `1bed2aa...`). `lint` command available in Makefile. Ruff configured in `pyproject.toml`.
*   **Type Checking (`mypy`):** *[Status: Not Implemented]* Type hints added to functions, but `mypy` is not yet installed or run as part of checks. *(Optional: Add mypy to dev dependencies and Makefile lint step)*.
*   **Makefile Documentation:** *[Status: Completed]* Targets have `## Help text` comments for self-documentation via `make help`.

---

**Final Steps Reminder for Team Zeal (Phase 1):**

*   Fill in the actual baseline accuracy achieved and link the WandB run in section 5.2.
*   Fill in the contribution details in section 3.2 (and README section 7).
*   Review and finalize docstrings/comments in the code (`dataset.py`, `train.py`).
*   (Optional) Add and run `mypy` checks.
*   (Recommended) Add the saved `best_model.pth` from a successful run to DVC tracking (`dvc add outputs/RUN_DIR/best_model.pth -o models/resnet18_baseline.pth`) and commit the new `.dvc` file.
*   Do one last `git add .`, `git commit -m "Finalize Phase 1 deliverables"`, and `git push`.
*   Ensure one team member submits the GitHub repository link on D2L.