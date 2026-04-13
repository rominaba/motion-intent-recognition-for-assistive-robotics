# Machine Learning-Based Motion Intent Recognition for Assistive Robotics

This project aims to enhance the motion intent recognition 
capability of assistive robots by processing and classifying data 
from wearable sensors. 

### Team Members

Michal Ridner  
Onur Ozkaya  
Romina Bahrami  

---

## Abstract

Assistive robotic systems such as wearable exoskeletons aim to improve mobility and independence for individuals with movement impairments. For these systems to provide effective support, they must accurately infer a user’s motion intent, such as walking, sitting, or standing, in real time and adapt their assistance accordingly. Traditional rule-based approaches often rely on predefined thresholds or heuristic rules applied to sensor signals, which may struggle to generalize across users and complex movement patterns. Machine learning offers a data-driven alternative that can learn patterns directly from sensor data and improve the accuracy and adaptability of motion intent recognition.

This project investigates the use of machine learning techniques to classify a set of human motion activities from wearable sensor data. An open-source Human Activity Recognition (HAR) dataset is used, containing three-dimensional linear acceleration and angular velocity measurements collected during various activities. The dataset is explored and preprocessed, and engineered features are extracted to represent the motion signals. The task is formulated as a multi-class classification problem.

Several machine learning models were considered, including Logistic Regression, Support Vector Classifier (SVC), Support Vector Machines (SVM), and Multilayer Perceptron (MLP). After Exploratory Data Analysis (EDA) and preliminary experiments, **logistic regression** was chosen as the primary approach, while an **MLP** in **PyTorch** (`notebooks/MLPClassifier.ipynb`) provides a **nonlinear** comparison with **grid-searched** hyperparameters. The **multinomial logistic regression** core is implemented **from scratch** in NumPy; sklearn implementations are used for additional baselines.
The final models are evaluated on a held-out test set and compared with trained versions of Scikit-learn's implementations where applicable. Performance is assessed using standard classification metrics, including accuracy, precision, recall, and F1-score.

---

## Data Source and Properties

The studied dataset is **UCI “Human Activity Recognition Using Smartphones”**, which must be downloaded and placed in `data/` directory prior to running any experiments.  
The data may be accessed through one of these links:    
[https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)  
[https://archive-beta.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive-beta.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)    
Data were collected from **30 volunteers** (ages 19–48) performing **six activities**: walking, walking upstairs, walking downstairs, sitting, standing, and laying. A smartphone (Samsung Galaxy S II) was worn on the **waist**. **Triaxial linear acceleration** and **triaxial angular velocity** were recorded at **50 Hz**. 

The authors cleaned up the data by removing noise and separating useful movement from background effects like gravity. Then, they split the data into small overlapping chunks about 2.5 seconds long to capture continuous activity without gaps. For each chunk, they calculated a large set of measurements (561 features) that describe the movement, looking at both how it changes over time (time-based) and the patterns or rhythms within it (frequency-based).
After analyzing the dataset and label distribution, we decided to use the precomputed feature matrices (`X_train.txt`, `X_test.txt`).

The benchmark partitions participants instead of rows at random; roughly 70% of subjects contribute to the training set and 30% to the test set. This design evaluates how well the models **generalize to new people**, which is important for assistive applications.

The training pipeline expects a root directory containing at least `train/X_train.txt`, `train/y_train.txt`, `test/X_test.txt`, `test/y_test.txt`, and `activity_labels.txt`. The training matrix has **7,352** rows and the test matrix **2,947** rows, each with **561** features.

According to the dataset documentation, feature values were **normalized and bounded** to approximately [-1, 1] by the dataset authors. The project’s preprocessing **additionally applies z-score standardization** (fit on training data only) before model fitting, which is appropriate for gradient-based logistic regression.

---

## Repository Content


| Location                            | Description                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/`                             | UCI-style HAR files: feature matrices, labels, subject IDs, feature names, activity names, and optional raw inertial streams.                                                                                                                                                                                                     |
| `src/models/preprocessing.py`       | Loads train/test features and labels, maps activity indices to names, **label-encodes** targets, and **standardizes** features with `StandardScaler`.                                                                                                                                                                             |
| `src/models/logistic_regression.py` | A **from-scratch multinomial logistic regression** in NumPy: softmax outputs, mean **cross-entropy** loss, full-batch **gradient descent**, optional early stopping when the loss change falls below a small threshold.                                                                                                           |
| `src/models/pca_reduction.py`       | Optional **PCA**: searches the number of components on a **stratified train/validation** split using a lightweight logistic-regression probe, prefers the smallest dimensionality that explains at least a target fraction of variance (default 90%), then refits PCA on the **full** training set and transforms train and test. |
| `src/utils.py`                      | Project paths (`PROJECT_ROOT`, `DATA_DIR`), logging, random seeds (including PyTorch for other experiments), device helpers, and `**evaluate_model`** (accuracy, macro precision/recall/F1, sklearn `classification_report`).                                                                                                     |
| `train_logistic_regression.py`      | Command-line entry point: load data → preprocess → optional PCA → optional **Optuna** hyperparameter search over learning rate and `max_iter` → train on full (possibly reduced) training set → report train/test metrics → save weights and metadata to `**.npz`**.                                                              |
| `notebooks/`                        | Exploratory analysis, baselines, extended logistic-regression experiments, feature ablation, and `**MLPClassifier.ipynb`** (PyTorch MLPs with grid search).                                                                                                                                                                       |
| `requirements.txt`                  | Python dependencies                                                                                                                                                                                                                                                                                                               |


---

## Exploratory Data Analysis

Exploratory work is documented primarily in `**notebooks/EDA.ipynb`**. The notebook:

- Loads **feature names**, **training features**, **activity labels**, **subject IDs**, and the **activity label dictionary**.
- Reports **dataset shape** (sample count × 561 features), lists the **six activity names**, and verifies **absence of missing values** in features, labels, and subject columns for the training split (and similarly for the test split where applicable).
- Visualizes **per-class frequencies** and **per-subject sample counts** to assess **class imbalance** (counts are on the same order across activities, with mild variation) and **subject imbalance** (some participants contribute more windows than others).
- Compares **label distributions between training and test** to detect **distribution shift**; it concludes that distributions are **consistent** across splits and, the official subject-wise train/test partitions are suitable for our study.
- Applies **standardization** and **PCA** for **2D/3D projections** and **explained-variance** curves, supporting intuition about redundancy and low-dimensional structure in the 561-dimensional space.

The EDA supports treating the problem as a **standard multiclass classification** task on **tabular features**, with no imputation or re-arranging splits required for our project.

---

## Modeling Approach

### Baseline and Primary Classifier: Multinomial Logistic Regression

The main classifier is **multinomial logistic regression** implemented **directly in NumPy** (`LogisticRegressionModel`). For input matrix X, weight matrix W, and bias vector b, the model computes class probabilities via **softmax**(X W^\top + b) and minimizes **average cross-entropy** over training samples using **Full-batch Gradient Descent**. Weights are initialized with small Gaussian noise; biases start at zero. The implementation exposes `**predict_proba`** and `**predict`** (argmax over classes). The vectorized implementation enables all samples to be processed simultaneously, improving efficiency during both training and inference.

#### Optional Dimensionality Reduction (PCA)

When enabled (e.g. via `--pca` on the training script), `**fit_best_pca_then_transform`** searches over candidate **principal component counts** using a **held-out validation** split stratified by class. A **probe** logistic regression is fit on PCA-transformed training folds; the procedure balances **validation accuracy** with a stopping rule tied to **cumulative explained variance** (default target 90%). The final PCA is **refit on all training samples** before transforming both train and test, avoiding test leakage.

#### Hyperparameter Tuning

`train_logistic_regression.py` optionally runs **Optuna** trials over `**learning rate`** (log-uniform range) and `**max_iter`**, maximizing **validation accuracy** on a stratified split. When tuning is disabled, fixed defaults (e.g. learning rate 0.05, `max_iter` 10,000) are used. To prevent overfitting and keep the training process more efficient, penalty rates for large `max_iter` and small `learning_rate`

#### Training Artifacts

Trained `**weight_matrix`**, `**bias_vector`**, and metadata (e.g. learning rate, `max_iter`, random state, target names) can be written to a **compressed `.npz`** file under a configurable path (defaulting under `checkpoints/`).

#### How to run the Training Script

Create a virtual environment, install dependencies, and run from the repository root:

```bash
pip install -r requirements.txt
python train_logistic_regression.py --data-path data
```

Useful flags include `**--pca**` (enable PCA search and transform before training), `**--n-trials N**` with `**N > 0**` to enable Optuna, `**--val-fraction**` for validation size, and `**--output-path**` for the saved `.npz`.

### Multilayer Perceptron (PyTorch, `notebooks/MLPClassifier.ipynb`)

As a **nonlinear** approach, the notebook implements a custom `**MLPClassifier`** (`torch.nn.Module`) with three depth presets:

- **Small:** 561 → 128 → 64 → 6 classes  
- **Medium:** 561 → 256 → 128 → 64 → 6  
- **Large:** 561 → 512 → 256 → 128 → 64 → 6

Each hidden block uses **ReLU**, **batch normalization**, and **dropout**; the **final two layers** omit batch norm and dropout (as in the notebook). Training uses **mini-batches**, the **Adam** optimizer (**weight decay** 5 \times 10^{-4}), and **cross-entropy with label smoothing (0.1)**. Each epoch updates on the training split and scores **validation accuracy**; **weights with the best validation accuracy are checkpointed** and restored after training.

The original training matrix is split **80% / 20%** with `**train_test_split`** (**stratified**, fixed random seed) into **train** and **validation** subsets. A **grid search** over **architecture size**, **learning rate**, **batch size**, and **short runs (15 epochs)** picks hyperparameters by **validation accuracy**. The chosen configuration is then trained **again for 30 epochs** with the same train/val split before evaluation on the **official UCI test set** (still subject-held-out). A second experiment applies `**fit_best_pca_then_transform`** (same PCA helper as logistic regression) and repeats the grid search and refit on **PCA-transformed** features.

---

## Evaluation

**Metrics.** Evaluation uses **accuracy**, **macro-averaged precision, recall, and F1**, and a per-class `**classification_report`**, via `src/utils.evaluate_model` and the baseline notebook.

---

## Results

Experiments use the **official subject-wise train/test split** (no row-level leakage), **z-score standardization** fit on training data, and the same **label encoding** across runs. Unless noted, metrics are **test-set** figures from the notebooks.

### Test-set comparison: all models (`baseline.ipynb` + `MLPClassifier.ipynb`)

Metrics are computed on the **official UCI subject-held-out test set** after **standardization** (fit on training data). **PCA** uses `**fit_best_pca_then_transform`** (**67** components, **~91%** variance explained in the logged runs). Rows from `**baseline.ipynb`** report **mean ± standard deviation** over **5** random seeds (Optuna on a stratified validation split per seed, then refit on the **full** training set). **MLP** rows report **mean ± standard deviation** over **5** seeds from `**MLPClassifier.ipynb`** (80/20 train/validation for grid search and checkpointing, then official test evaluation).


| Model                        | Accuracy            | Macro precision     | Macro recall        | Macro F1            |
| ---------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Decision tree (all features) | 0.858975 ± 0.004229 | 0.858413 ± 0.004546 | 0.855565 ± 0.004366 | 0.855948 ± 0.004279 |
| sklearn’s LR (all features)  | 0.952019 ± 0.001837 | 0.955530 ± 0.001512 | 0.950909 ± 0.001811 | 0.952114 ± 0.001770 |
| Our custom LR (all features) | 0.944079 ± 0.000736 | 0.947411 ± 0.001071 | 0.943112 ± 0.000764 | 0.944330 ± 0.000784 |
| Our MLP (all features)       | 0.957245 ± 0.006697 | 0.960038 ± 0.005185 | 0.956407 ± 0.007116 | 0.957320 ± 0.006817 |
| Decision tree (after PCA)    | 0.761452 ± 0.002736 | 0.762879 ± 0.003138 | 0.757075 ± 0.002741 | 0.758662 ± 0.002772 |
| sklearn’s LR (after PCA)     | 0.924669 ± 0.001336 | 0.925365 ± 0.001386 | 0.922977 ± 0.001222 | 0.923754 ± 0.001267 |
| Our custom LR (after PCA)    | 0.922362 ± 0.001387 | 0.923287 ± 0.001233 | 0.920260 ± 0.001461 | 0.921230 ± 0.001406 |
| Our MLP (after PCA)          | 0.922701 ± 0.006194 | 0.924038 ± 0.005444 | 0.920419 ± 0.006911 | 0.921454 ± 0.006659 |


Note: The **MLP notebook** trains on **80%** of the training rows and uses **20%** for **validation and checkpointing**; `**baseline.ipynb`** refits on the **full** training matrix after tuning each seed. Numbers are therefore **not from an identical training-data protocol**; the MLP result is still a fair **held-out subject test** evaluation, but **direct comparison** to the baseline table should keep this split difference in mind.

The linear models outperform the **full-feature** decision tree by a wide margin; **our custom LR** is within about **0.8** percentage points of sklearn on mean accuracy and macro F1. **Our MLP** (full features) is **competitive with** sklearn’s logistic regression on mean test accuracy in this run. **PCA** hurts the **tree** most; **logistic** models and the **MLP** each lose a few points versus full features; under PCA the **MLP** remains **in line with** the stronger linear baselines on mean metrics. Confusion persists between **walking** and **walking upstairs**, and between **sitting** and **standing**, as discussed in the notebooks.

### Weight-based analysis (`notebooks/analyze_predictions.ipynb`)

The notebook inspects a **saved checkpoint** of the trained multinomial logistic model (softmax weights and biases). **Global importance** (magnitude of weights across classes) is dominated by **gravity-acceleration statistics** (means, min/max, energy) and **angles to the gravity mean**, together with **gyroscope entropy** features. **Per-class** patterns show, for example, strong **orientation-related** weights for static activities (sitting, standing, laying) and **correlation / jerk / frequency-energy** terms for walking and stair activities. **Bias terms** are small in magnitude, so the model does not rely on a strong default class prior.

### Feature ablation (`notebooks/feature_ablation.ipynb`)

The same **custom logistic regression** is trained on **feature subsets** defined by name patterns in `features.txt`. Approximate **test accuracies**:


| Subset                                   | Test accuracy |
| ---------------------------------------- | ------------- |
| Gyroscope only                           | **81.1%**     |
| Accelerometer only                       | **89.7%**     |
| Body acceleration only                   | **91.5%**     |
| Gravity only                             | **86.0%**     |
| Time domain (`t`*) only                  | **94.6%**     |
| Frequency domain (`f`*) only             | **87.5%**     |
| X-axis only                              | **91.2%**     |
| Y-axis only                              | **82.2%**     |
| Z-axis only                              | **76.6%**     |
| Multi-axis (combined-axis) features only | **89.4%**     |


---

## Discussion

Overall performance On the UCI smartphone HAR benchmark with **new subjects** in the test set, **strong linear models** (sklearn and custom logistic regression) reach about **94–95% test accuracy** and **~0.94–0.95 macro F1** (`baseline.ipynb`). The **PyTorch MLP** in `MLPClassifier.ipynb` lands in the **same accuracy band**; so **extra depth and nonlinearities do not clearly beat** a well-tuned **linear softmax** on these **engineered features**. The **custom NumPy logistic regression** remains **slightly below** sklearn and the MLP on accuracy. The **custom logistic regression** still **matches sklearn closely**, which validates the implementation and preprocessing.

Why the decision tree lags:
The **decision tree** likely **underfits or splits less effectively** in a **high-dimensional, correlated** space compared to **dense linear boundaries** learned by logistic regression. After **PCA**, the tree’s accuracy **falls sharply**, while logistic regression **degrades only slightly**—suggesting that **a low-dimensional linear subspace** captures most of the label-relevant variation for a **linear** decision rule, but **not** the kind of **piecewise rules** the tree used in the original feature basis.

PCA:
Retaining **67** components (**~91%** variance) shrinks the input by roughly an **8×** factor versus 561 raw features. **Logistic regression** still reaches **~92–93%** test accuracy—only a few points below the full-feature models—so most of what a **linear** classifier needs is already captured in a **compact** variance-preserving subspace. That makes PCA a plausible **trade-off** when **memory, bandwidth, or inference cost** matter, accepting a small accuracy gap. The **decision tree** does not benefit PCA. The projection **mixes** original axes, which **weakens axis-aligned splits** and drives accuracy to **~76%**, so **PCA is a poor pairing** with this tree baseline. In short, PCA here behaves like **linear compression** that **preserves softmax-friendly structure** but **disrupts interpretable, coordinate-wise** rules.
For the MLP, **PCA + refit** **reduces** mean test accuracy versus full features, but by a **smaller** margin than for the decision tree. The relative ranking of models under PCA can differ from the full-feature case; the notebooks discuss confusion patterns and architecture choices in more detail.

Interpretability and physiology:
The logistic regression weights show that the model distinguishes activities using both movement patterns and body orientation relative to gravity. Dynamic activities such as walking, walking upstairs, and walking downstairs are mainly identified through features that capture differences in rhythm, impact, and coordination. For example, walking is associated with more regular and coordinated motion, while upstairs and downstairs movements show stronger gravity and energy-related patterns. Sitting and standing are characterized by stable posture-related signals, while laying is most strongly defined by a different alignment of the body relative to gravity. Overall, the results suggest that the model is learning meaningful physical patterns.

Ablation insights:
**Time-domain features alone** nearly match the **full** feature set, so **most discriminative signal** for this model lives in **time statistics** rather than **frequency-only** views. **Frequency-only** inputs **lose clarity on static classes**, which the notebook attributes to **weaker orientation/gravity cues** in that subset. **Gyro-only** data struggles to separate **similar rotational patterns** across walking variants and **low-motion** static classes; **accelerometer-only** data still confuses **walking directions** and **sitting vs. standing**, where **gravity and subtle motion** together matter. **Body-acceleration** features achieve **strong dynamic-activity** scores but still **confuse static** classes without **full orientation** context, while **gravity-only** features separate **dynamic vs. static** and **laying** well but **not fine-grained walking** or **sitting vs. standing**. **Single-axis** subsets show **X** as most informative among the three phone axes in isolation, while **Z** is weakest—consistent with **how much each axis encodes** the activities in this mounting. Together, the ablations support the design of the **full 561-feature** vector: **complementary modalities and domains** are needed for **robust** subject-general recognition.

---

## 8. Limitations and Potential Next Steps

Results apply to the **UCI smartphone HAR** setting; assistive robots may use different sensor placements, sampling rates, or labels. **Domain adaptation**, **recalibration**, or **collection of robot-relevant data** would be needed for better generalization.

The official **subject-held-out test set** is appropriate for reporting, but **nested or grouped cross-validation by subject** on the training pool could yield **tighter confidence intervals** and better model comparison.

---

## Citation of the Dataset

If this work is published, the **UCI HAR dataset** should be cited according to the instructions in `data/README.txt` (see the reference to Anguita et al., ESANN 2013).