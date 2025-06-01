# PHASE 1: Project Design & Model Development

## 1. Project Proposal

### **1.1 Project Scope and Objectives**

* **Problem Statement:** Develop an MLOps pipeline capable of training an image classifier, detecting performance degradation due to common data corruptions (drift), and automatically triggering retraining to maintain accuracy.
* **Project Objectives:**
    1. Achieve a baseline classification accuracy of ≥ 85% on the clean Imagenette-160 validation split using a fine-tuned ResNet-18 model. *(Status: Achieved - 96.76%)*
    2. Implement a monitoring system capable of reliably detecting simulated drift scenarios (e.g., noise, brightness shifts) based on statistical tests or pre-defined thresholds. *(Note: Drift detection implementation is planned for Phase 2).*
    3. Establish an automated retraining pipeline triggered by the detection of significant data drift. *(Note: Automation implementation is planned for Phase 2/3).*
* **Success Metrics:**
  * Model Performance: Top-1 Classification Accuracy (Baseline: 96.76% achieved in Phase 1).
  * Drift Detection: Drift Detection Rate / F1-score on simulated drift data (To be evaluated in Phase 2).
  * Pipeline Automation: Successful execution rate of the automated retraining workflow (To be evaluated in Phase 2/3).

### **1.2 Selection of Data**

* **Dataset(s) chosen and justification:**
  * Selected **Imagenette-160 (v2 split)**. Contains ~13k train/~5k val images across 10 ImageNet classes.
  * **Justification:** Manageable size (~102MB download size), realistic 160px resolution suitable for pre-trained models (`timm`) and meaningful drift simulation compared to lower-res alternatives like CIFAR-10.
* **Data source(s) and access method:**
  * Downloaded from the [FastAI GitHub repository](https://github.com/fastai/imagenette) (`.tar.gz` archive).
  * Raw data (`imagenette2-160.tgz`) versioned using DVC with Google Drive as remote storage. Metafile: `data/raw/imagenette2-160.tgz.dvc`.
  * Accessed in code via `torchvision.datasets.ImageFolder` after extraction by `drift_detector_pipeline/dataset.py` (run via `make process_data`).
* **Preprocessing steps:**
  * **Implemented:** Extraction of `.tar.gz` archive via `dataset.py`. Standard image transformations (Resize(256), CenterCrop(224) for val / RandomResizedCrop(224) + RandomHorizontalFlip for train, `ToTensor`, ImageNet normalization) applied via `torchvision.transforms` within the `DataLoader` setup (`get_transforms` function in `dataset.py`).
  * **Planned (Phase 2):** Integration of drift-specific transformations (e.g., brightness, noise) configurable via Hydra.

### **1.3 Model Considerations**

* **Chosen Model Architecture:** **ResNet-18**
* **Rationale for Model Choice:** Strong baseline performance, computational efficiency suitable for course hardware/Colab, excellent pre-trained weights available via `timm`.
* **Architectural Summary:** Standard ResNet-18 architecture with residual blocks ([He et al., 2015](https://arxiv.org/abs/1512.03385)).
* **Source/Implementation:** Utilized `timm.create_model('resnet18', pretrained=True)` ([timm GitHub Repo](https://github.com/huggingface/pytorch-image-models)) via `train.py`, replacing the final classifier head for 10 classes (specified in `conf/config.yaml`).
* **Implementation (Phase 1):** Loaded pre-trained model, adapted classifier, implemented basic fine-tuning loop (optimizer: AdamW, scheduler: CosineAnnealingLR configured via Hydra), integrated experiment tracking (WandB). Saved best model based on validation accuracy. *(Status: Completed)*

### **1.4 Open-source Tools**

* **Third-Party Package Selected:** `timm` (PyTorch Image Models)
* **Brief description of how/why used:** Selected `timm` ([timm GitHub Repo](https://github.com/huggingface/pytorch-image-models)) for its easy API (`timm.create_model`) to load high-quality pre-trained ResNet-18 weights for transfer learning on Imagenette-160, as implemented in `train.py`.

## 2. Code Organization & Setup

### **2.1 Repository Setup**

* **GitHub Repository:** Central repository created at: [https://github.com/estmon8u/team-zeal-project](https://github.com/estmon8u/team-zeal-project)
* **Project Structure:** Standard structure initialized using Cookiecutter Data Science template, adapted for this project (see `README.md` for layout). Includes directories for source code (`drift_detector_pipeline/`), data (`data/`), models (`models/`), config (`conf/`), tests (`tests/`), docs (`docs/`) and key files (`Makefile`, `.gitignore`, `pyproject.toml`). *(Status: Completed)*

### **2.2 Environment Setup**

* **Python Virtual Environment:** Managed using `venv` and `pip`. Activated via `source .venv/bin/activate` (Mac/Linux) or equivalent (Windows). *(Status: Completed)*
* **Dependency Management:** Dependencies defined in `pyproject.toml` and installed via `pip install -e .`. A full list including transitive dependencies is in `requirements.txt`. Key MLOps tools (`dvc[gdrive]`, `hydra-core`, `wandb`, `ruff`, `timm`, `pytest`) are included. *(Status: Completed)*
* **Development Platform:** Google Colab used for GPU-accelerated training runs; local environment used for code development and testing. *(Status: Completed)*

## 3. Version Control & Collaboration

### **3.1 Git Usage**

* **Repository:** All code, configuration (`conf/config.yaml`), documentation (`README.md`, `PHASE1.md`), and DVC meta-files (`.dvc/config`, `data/raw/imagenette2-160.tgz.dvc`, `models/resnet18_baseline_v1.pth.dvc`) tracked in the GitHub repository linked above. *(Status: Completed)*
* **Commit Practices:** Followed standard practices for frequent, atomic commits with descriptive messages (e.g., `ac199a43e95ef05f0a80ad7dbb511eaa6458e127` - "Adding simple unit tests"). *(Status: Completed)*
* **Branching Strategy:** Development occurred on `main` and `develop` branches during Phase 1, with merges into `main`. *(Status: Completed)*
* **Merging:** Merges from `develop` to `main` performed. Pull Requests planned for future feature integration. *(Status: Completed)*

### **3.2 Team Collaboration**

* **Roles (Phase 1 - Inferred):**
  * Esteban Montelongo: DVC setup & data versioning, `dataset.py` (extraction, transforms, dataloaders), initial documentation structure (`README.md`, `PHASE1.md`), architecture diagram, model DVC tracking, unit test implementation.
  * Sajith Bandara: Hydra integration (`conf/config.yaml`, `train.py` decorator/config usage), `train.py` core structure (model loading, optimizer, scheduler, loop), Makefile setup (`train`, `process_data` rules), model saving path correction.
  * Arjun Kumar Sankar Chandrasekar: WandB integration (`wandb.init`, `wandb.log`), dependency management (`pyproject.toml`, `requirements.txt`), `ruff` configuration and code formatting, testing infrastructure setup and test contributions.
* **Communication:** Primary communication via MS Teams channel 'Team Zeal SE489'. *(Status: Ongoing)*
* **Code Reviews:** Conducted informally for Phase 1 setup. Formal Pull Request reviews planned for Phase 2 onwards. *(Status: Informal completed)*
* **WandB Team:** Collaboration established using the `emontel1-depaul-university` WandB entity. Teammates invited and can view/log runs. *(Status: Completed)*

## 4. Data Handling

### **4.1 Data Preparation**

* **Data Acquisition & Versioning:** Imagenette-160 (`.tar.gz`) downloaded to `data/raw/`, tracked by DVC (metafile `data/raw/imagenette2-160.tgz.dvc`), and pushed to the configured Google Drive remote. *(Status: Completed)*
* **Preprocessing Script:** Raw data extraction logic implemented in `drift_detector_pipeline/dataset.py` (`extract_data` function), runnable via `make process_data`. *(Status: Completed)*
* **Dataset Loading:** PyTorch `DataLoaders` implemented in `drift_detector_pipeline/dataset.py` (`get_dataloaders` function) using `torchvision.datasets.ImageFolder`, applying transformations defined in `get_transforms`. *(Status: Completed)*

### **4.2 Data Documentation**

* **README:** Main `README.md` includes Setup/Usage sections. *(Status: Completed)*
* **Script Documentation:** Docstrings and comments added to `dataset.py` and `train.py`. *(Status: Completed)*

## 5. Model Training

### **5.1 Training Infrastructure**

* **Platform:** Baseline training executed successfully on Google Colab GPU / Local CPU. *(Status: Completed)*
* **Environment:** Setup reproducible via cloning repo and `pip install -e .`. *(Status: Completed)*
* **Configuration:** Managed centrally via Hydra (`conf/config.yaml`). *(Status: Completed)*
* **Experiment Tracking:** Weights & Biases (WandB) used for logging metrics, hyperparameters. Runs logged to project `zeal-imagenette-drift` under entity `emontel1-depaul-university`. *(Status: Completed)*

### **5.2 Initial Model Training & Evaluation**

* **Training Script:** Implemented in `drift_detector_pipeline/modeling/train.py`, runnable via `make train`. *(Status: Completed)*
* **Baseline Training:** Initial training run executed for 10 epochs. Baseline validation accuracy goal (≥85%) **achieved**, reaching **96.76%**. *(Status: Completed)*
* **WandB Run Link:** [https://wandb.ai/emontel1-depaul-university/zeal-imagenette-drift](https://wandb.ai/emontel1-depaul-university/zeal-imagenette-drift) *(Placeholder - Insert direct link to the specific run if available)*
* **Evaluation:** Top-1 Accuracy on validation set tracked per epoch via WandB. *(Status: Completed)*
* **Model Saving & Versioning:** Best model checkpoint (based on validation accuracy) saved automatically by `train.py` to the Hydra output directory. This artifact was then manually added to DVC tracking as `models/resnet18_baseline_v1.pth` (metafile `models/resnet18_baseline_v1.pth.dvc`) and pushed to the remote. *(Status: Completed)*

## 6. Documentation & Reporting

### **6.1 Project README**

* **Status:** *[Status: Updated]* Main `README.md` updated with setup, usage, architecture diagram, contribution summary, tool list, and Phase 1 status.

### **6.2 Code Documentation**

* **Docstrings:** *[Status: Completed]* Added to key functions in `dataset.py` and `train.py`.
* **Inline Comments:** *[Status: Completed]* Added for clarity in scripts.
* **Code Styling & Formatting (`ruff`):** *[Status: Completed]* Code formatted using `ruff format`. `make lint` and `make format` commands available in Makefile.
* **Type Checking (`mypy`):** *[Status: Not Implemented]* Type hints added, but `mypy` checks not yet integrated into CI/workflow.
* **Makefile Documentation:** *[Status: Completed]* Targets have `## Help text` comments.
* **Testing:** *[Status: Completed]* Unit tests implemented in `tests/test_data.py` covering data extraction and transformation logic using `pytest` and `unittest.mock`. Tests can be run via `make test`.

---

**Phase 1 Completion Checklist:**

* [X] Baseline training run completed and results logged.
* [X] Model performance meets/exceeds target (96.76% > 85%).
* [X] Code, config, DVC meta-files committed to Git.
* [X] Raw data versioned with DVC and pushed to remote.
* [X] Baseline model artifact versioned with DVC and pushed to remote.
* [X] WandB collaboration set up and run logged to shared entity.
* [X] Code formatting applied (`ruff format`).
* [X] Documentation (`README.md`, `PHASE1.md`) updated with results and status.
* [X] Implement and pass basic unit tests.
* [X] Review of code documentation (docstrings, comments).

---
