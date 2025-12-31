# Emotion-Aware Burnout Indicator

## Project Overview

This repository documents an applied research project examining emotion-informed burnout indicator modelling using baseline and transformer-based natural language processing models, grounded in a literature-based interpretive framework.

The project includes exploratory data analysis, data preprocessing, baseline modelling, transformer-based experimentation, and evaluation using the publicly available **GoEmotions dataset**. While caregiving-related emotional burden motivates the research, the study adopts a domain-agnostic approach and does not rely on caregiver-specific textual data.

A lightweight web-based prototype is also developed to demonstrate the integration of the trained emotion classification model with a literature-informed affective mapping framework.

---

## Quick Start

1. Upload the `applied_research_project/` folder to Google Drive (under **MyDrive**).
2. Open `01_eda_data_cleaning.ipynb` in Google Colab.
3. Mount Google Drive using:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Run notebooks sequentially as documented in the **Notebook Execution Order** section.

---

## Dataset

This project uses the **GoEmotions** dataset, a publicly available, human-annotated emotion dataset released by Google Research. It contains textual data labelled with fine-grained emotions and is widely used in emotion classification research.

The dataset was obtained from the official Google Research repository and includes predefined training, validation, and test splits, which were used as provided.

Raw textual data is not included due to licensing constraints. Preprocessed data are provided under the `data/` directory.

---

## Execution Environment

This project was developed and executed using **Google Colab Pro**, with GPU acceleration used for transformer-based experiments. This environment was selected to ensure:

- Access to GPU acceleration for transformer fine-tuning
- Sufficient memory and runtime stability
- A consistent and reproducible execution environment

All experiments, results, and saved artefacts correspond to execution within this environment.

---

## Google Drive Integration

Google Drive is mounted within the Colab runtime to ensure that data and trained models persist across sessions. This creates a Colab-specific mount point at `/content/drive`, allowing direct access to files stored in Google Drive.

---

## Project Directory Structure and Paths 

The project is organised within a top-level folder named: 
- **Project root:** `applied_research_project/`

When executed in Google Colab, this folder is expected to be located under **MyDrive** and accessed via:
- `/content/drive/MyDrive/applied_research_project/`

All notebooks assume this directory structure. The following subdirectories are used throughout the project: 

- **Cleaned data directory:**  
  `/content/drive/MyDrive/applied_research_project/data`

- **Prototype directory:**  
  `/content/drive/MyDrive/applied_research_project/burnout_indicator_detection_prototype`

- **Best model storage path:**  
  `/content/drive/MyDrive/applied_research_project/burnout_indicator_detection_prototype/best_emo_model`

If the project folder is placed in a different location, the paths should be updated accordingly. Execution outside Google Colab may require minor path and environment configuration adjustments.

---

## Project Contents

### Core Notebooks

The following notebooks constitute the primary pipeline used for analysis and evaluation:

- **01_eda_data_cleaning.ipynb:**  
  Exploratory data analysis and data cleaning

- **02_baseline_models.ipynb:**  
  Baseline machine learning models

- **03_transformers.ipynb:**  
  Transformer-based model training and burnout indicator mapping

- **streamlit_prototype.ipynb:**  
  Generates `inference.py` and `app.py` for the prototype execution pipeline and Streamlit-based user interface

### Supplementary Notebooks

The following notebooks contain additional experiments conducted during model development and analysis:

- **supplementary_experiment_transformers_emoji_emoticon_handling.ipynb:**  
  Transformer experiments incorporating emoji and emoticon processing

- **supplementary_experiment_alternative_data_split_75_15_10.ipynb:**  
  Additional experiment using a 75/15/10 (train/validation/test) split

- **supplementary_exploratory_baseline_and_transformer_experiments.ipynb:**  
  Exploratory and development-stage experiments, including baseline solver selection, thresholding approaches, and pooling methods used for quadrant aggregation. These experiments      informed the final design choices implemented in the core notebooks.

---

## Supporting Files and Directories

- **Shared utility script: `utils.py`**  
  Contains shared functions for emotion categorisation, probability aggregation, threshold tuning, and model evaluation. This file is initially created in `01_eda_data_cleaning.ipynb` and extended in `02_baseline_models.ipynb`, with the consolidated utilities reused across subsequent notebooks. Centralising this logic reduces code duplication and improves consistency across experiments.

- **Preprocessed data: `data/`**  
  Contains preprocessed feature representations and labels used across experiments for both baseline and transformer-based models.

- **Prototype components: `burnout_indicator_detection_prototype/`**  
  Contains the files required to run the web-based prototype. Due to size constraints, the saved best-performing transformer model is not included in the repository. When running `03_transformers.ipynb`, the best model is automatically saved under:  
  `burnout_indicator_detection_prototype/best_emo_model/`

---

## Notebook Execution Order

The notebooks are intended to be run in the following order:

1. `01_eda_data_cleaning.ipynb`
2. `02_baseline_models.ipynb`
3. `03_transformers.ipynb`
4. `streamlit_prototype.ipynb`

After running the core notebooks, the supplementary notebooks can be executed in any order.

---

## Runtime Considerations

- Data cleaning and baseline model notebooks can be executed on **CPU**
- Transformer-based training and evaluation notebooks may require extended runtime and are recommended to be run with **GPU acceleration**

---

## Dependency Management

All required Python dependencies are installed directly within the notebooks using `pip`, following standard Google Colab practice. This setup supports reproducible execution within the Google Colab environment.

---

## Reproducibility

This project is fully reproducible within the Google Colab environment using the provided notebooks and documented directory structure.

---

## Ethical Note

This project is intended for academic research and technical demonstration only. It does not perform clinical diagnosis, assess mental health status, or provide decision support. Outputs generated by the system are illustrative and should not be used for evaluation, judgement, or real-world decision-making.


