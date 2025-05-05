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
    *   **Justification:** Manageable size (~102MB), realistic 160px resolution suitable for pre-trained models (`timm`) and meaningful drift simulation compared to lower-res alternatives like CIFAR-10.
*   **Data source(s) and access method:**
    *   Downloaded from the [FastAI GitHub repository](https://github.com/fastai/imagenette) (`.tar.gz` archive).
    *   Raw data versioned using DVC with Google Drive as remote storage.
    *   Accessed in code via `torchvision.datasets.ImageFolder` after extraction.
*   **Preprocessing steps:**
    *   **Implemented:** Extraction of `.tar.gz` archive via script (`dataset.py`). Resize (256px), Center/Random Crop (224px), `ToTensor`, ImageNet normalization applied via `torchvision.transforms` in the `DataLoader`.
    *   **Planned (Phase 2):** Integration of drift-specific transformations (e.g., brightness, noise) configurable via Hydra.

### **1.3 Model Considerations**

*   **Chosen Model Architecture:** **ResNet-18**
*   **Rationale for Model Choice:** Strong baseline performance, computational efficiency suitable for course hardware, excellent pre-trained weights available via `timm`.
*   **Architectural Summary:** Standard ResNet-18 architecture with residual blocks ([He et al., 2015](https://arxiv.org/abs/1512.03385)).
*   **Source/Implementation:** Utilized `timm.create_model('resnet18', pretrained=True)` ([timm GitHub Repo](https://github.com/huggingface/pytorch-image-models)), replacing the final classifier head for 10 classes.
*   **Implementation Plan (Phase 1):** Loaded pre-trained model, adapted classifier, implemented basic fine-tuning loop (optimizer: AdamW, scheduler: CosineAnnealingLR), integrated experiment tracking (WandB). *Status: Core logic implemented in `train.py`.*

### **1.4 Open-source Tools**

*   **Third-Party Package Selected:** `timm` (PyTorch Image Models)
*   **Brief description of how/why used:** Selected `timm` ([timm GitHub Repo](https://github.com/huggingface/pytorch-image-models)) for its easy API (`create_model`) to load high-quality pre-trained ResNet-18 weights for transfer learning on Imagenette-160.

## 2. Code Organization & Setup

### **2.1 Repository Setup**

*   **GitHub Repository:** Central repository created at: [GitHub - estmon8u/team-zeal-project](https://github.com/estmon8u/team-zeal-project)
*   **Project Structure:** Standard structure generated using Cookiecutter Data Science template, including directories like `data/`, `models/`, `drift_detector_pipeline/`, `conf/`, `tests/`, `docs/` and key files (`Makefile`, `.gitignore`, `pyproject.toml`).

### **2.2 Environment Setup**

*   **Python Virtual Environment:** Managed using `venv` and `pip`. Activated via `source .venv/bin/activate` or equivalent.
*   **Dependency Management:** Dependencies defined in `pyproject.toml` and installed via `pip install -e .[dev]`. Reproducible list generated in `requirements.txt`.
*   **Development Platform:** Google Colab used for GPU-accelerated training runs; local environment used for code development and testing.

## 3. Version Control & Collaboration

### **3.1 Git Usage**

*   **Repository:** All code, configuration, documentation, and DVC meta-files tracked in the GitHub repository linked above.
*   **Commit Practices:** Followed standard practices for frequent, atomic commits with descriptive messages.
*   **Branching Strategy:** Development occurred primarily on `main` for Phase 1 setup, with branching planned for future features/experiments.
*   **Merging:** Pull Requests to be used for feature integration in later phases.

### **3.2 Team Collaboration**

*   **Roles (Initial):** *[Example - Replace with your actual roles]*
    *   Esteban Montelongo: DVC setup, Data extraction/loading script (`dataset.py`).
    *   Sajith Bandara: Hydra setup (`config.yaml`), Training script structure (`train.py`), Makefile setup.
    *   Arjun Kumar Sankar Chandrasekar: WandB integration, `pyproject.toml`/`requirements.txt` management, README structure.
*   **Communication:** *(Example - Replace)* Primary communication via MS Teams channel 'Team Zeal SE489'. Weekly brief sync meetings held.
*   **Code Reviews:** Conducted informally during pair programming or via discussion for Phase 1 setup. Formal PR reviews planned for later phases.
*   **Merge Conflict Resolution:** N/A for Phase 1 (primarily worked on `main` or initial setup). Strategy defined for future phases.

## 4. Data Handling

### **4.1 Data Preparation**

*   **Data Acquisition & Versioning:** Imagenette-160 (`.tar.gz`) downloaded to `data/raw/`, tracked by DVC (metafile `data/raw/imagenette2-160.tgz.dvc` committed), and pushed to Google Drive remote.
*   **Preprocessing Script:** Logic for extracting the archive is implemented in `drift_detector_pipeline/dataset.py` (`extract_data` function), runnable via `make process_data`.
*   **Dataset Loading:** PyTorch `DataLoaders` implemented in `drift_detector_pipeline/dataset.py` (`get_dataloaders` function) using `torchvision.datasets.ImageFolder` on `data/processed/` with appropriate transforms defined in `get_transforms`, configured via Hydra.

### **4.2 Data Documentation**

*   **README:** Placeholder `data/README.md` exists (or content moved to main README).
*   **Script Documentation:** *[Status: To be completed/verified]* Docstrings and comments need to be added/finalized for `dataset.py`.

## 5. Model Training

### **5.1 Training Infrastructure**

*   **Platform:** Baseline training executed successfully on [Specify: Google Colab GPU / Local CPU / Local GPU].
*   **Environment:** Setup via cloning repo and `pip install -e .[dev]`.
*   **Configuration:** Managed via Hydra (`conf/config.yaml`).
*   **Experiment Tracking:** Weights & Biases (WandB) used; runs logged to project `zeal-imagenette-drift` under entity `emontel1-depaul-university`.

### **5.2 Initial Model Training & Evaluation**

*   **Training Script:** Implemented in `drift_detector_pipeline/modeling/train.py`, integrating Hydra, WandB, `timm`, and data loaders.
*   **Baseline Training:** *[Status: Completed/To be finalized]* Initial training run executed. [Mention achieved baseline validation accuracy, e.g., "Achieved X.XX% validation accuracy after Y epochs." Link to WandB run if possible].
*   **Evaluation:** Top-1 Accuracy on validation set tracked per epoch via WandB.
*   **Model Saving:** Best model checkpoint saved to Hydra output directory (e.g., `outputs/.../best_model.pth`). Model artifact versioned using DVC.

## 6. Documentation & Reporting

### **6.1 Project README**

*   **Status:** *[Status: To be completed/verified]* Main `README.md` requires finalization: project description, architecture diagram, updated setup/usage, contribution summary.

### **6.2 Code Documentation**

*   **Docstrings:** *[Status: To be completed/verified]* Need to be added/finalized for key functions/classes.
*   **Inline Comments:** *[Status: To be completed/verified]* Need to be added/reviewed for clarity.
*   **Code Styling & Formatting (`ruff`):** *[Status: To be run/verified]* Code needs final formatting and linting check using `ruff`.
*   **Type Checking (`mypy`):** *[Status: To be run/verified]* Type hints need to be added/verified; code needs to pass `mypy` check.
*   **Makefile Documentation:** *[Status: To be completed/verified]* Targets need `## Help text` comments added/finalized.

---

**Final Steps Reminder for Team Zeal:**

*   Fill in all `[Placeholders]` or `[Status: ...]` sections above.
*   Run the training (`make train`) and note the results.
*   Version the final model using DVC (`dvc add models/...`, `dvc push`).
*   Complete the Code Quality checks (`ruff`).
*   Finalize all documentation (`README.md`, code docstrings, Makefile comments).
*   Do one last `git add .`, `git commit`, and `git push`.
*   Submit the GitHub link on D2L.