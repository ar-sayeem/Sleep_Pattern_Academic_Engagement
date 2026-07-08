> [!IMPORTANT]
> **Peer Review Notice**
>
> This repository is made publicly available **solely for academic peer review and editorial evaluation** of the associated research manuscript.
>
> This repository is **NOT open source**.
>
> All source code, datasets, models, documentation, and other contents are protected by copyright.
>
> Except as expressly permitted under the **Academic Peer Review License (APRL) v1.0**, no permission is granted to copy, modify, redistribute, reuse, publish, or incorporate any portion of this repository into other research, software, datasets, products, or AI/ML systems.


# Sleep Pattern and Academic Performance Prediction
### Machine Learning Analysis Using GAN-Augmented Data

## 📋 Project Overview

This research project develops machine learning models to predict student CGPA based on sleep patterns and academic behavior. The project uses GAN-augmented balanced datasets and compares multiple classification approaches including traditional ML, ensemble methods, and deep learning.

---

## 🛠️ **Tools & Technologies Used**

### **Data Processing & Analysis**
- **Pandas** - Data manipulation and CSV processing
- **NumPy** - Numerical computations
- **Python 3.x** - Primary programming language

### **Machine Learning Models**
- **Scikit-Learn** - Traditional ML algorithms
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Cross-validation & metrics
  
- **XGBoost** - Optimized gradient boosting
  - XGBClassifier for multi-class classification
  - Advanced regularization parameters
  
- **CatBoost** - Categorical gradient boosting
  - CatBoostClassifier
  - Better handling of categorical features
  
- **PyTorch** - Deep learning framework
  - Neural network implementation
  - GPU/CUDA support
  - Custom model architecture with dropout regularization

### **Ensemble & Model Interpretation**
- **VotingClassifier** - Hard and soft voting ensembles
- **StackingClassifier** - Meta-learner based ensembling
- **SHAP** (SHapley Additive exPlanations) - Model interpretability
  - Global feature importance
  - Local prediction explanations
  - Dependence plots
  - Per-class analysis

### **Visualization & Plotting**
- **Matplotlib** - General plotting library
- **Seaborn** - Statistical visualization
  - Confusion matrices
  - Heatmaps
  - Distribution plots

### **Data Splitting & Metrics**
- **train_test_split** - 80-20 stratified split
- **Evaluation Metrics:**
  - Accuracy Score
  - Precision, Recall, F1-Score (macro-averaged)
  - Balanced Accuracy
  - ROC-AUC (One-vs-Rest, macro-averaged)
  - Confusion Matrix
  - Classification Report

<!-- ---

## 📁 **Folder Structure**

```
SleepPattern/
│
├── data/
│   ├── encoding.txt                          # Feature encoding definitions
│   ├── Final_Combined_Normalized.csv         # GAN-augmented balanced dataset (n=1918)
│   ├── Final_Decoded.csv                     # Decoded version
│   ├── Final_Encoded.csv                     # Encoded version
│   ├── Final_Encoded_Normalized.csv          # Encoded & normalized
│   └── Synthetic_Minority_Normalized.csv     # Synthetically generated data
│
├── notebooks/
│   └── 06_Balanced_Data_Model_Experiments_CGPA3.ipynb
│       Main experiment notebook with:
│       • 6 classification models
│       • 9 ensemble techniques
│       • SHAP interpretability analysis
│       • Feature importance visualization
│       • Per-class performance analysis
│
├── images/
│   ├── best_ensemble_confusion_matrix.png
│   ├── best_ensemble_top3_confusion_matrix.png
│   ├── comprehensive_model_comparison.png
│   ├── ensemble_comparison.png
│   ├── ensemble_top3_comparison.png
│   ├── feature_importance_5class_all_models.png
│   ├── feature_importance_5class_heatmap.png
│   ├── feature_importance_all_models.png
│   ├── model_comparison_balanced_normalized.png
│   ├── shap_global_importance_xgboost.png
│   ├── shap_beeswarm_xgboost.png
│   ├── shap_waterfall_*.png (individual explanations)
│   ├── shap_dependence_plots_xgboost.png
│   └── shap_per_class_comparison_xgboost.png
│
└── results/
    ├── balanced_normalized_model_results.csv
    ├── final_model_comparison.csv
    ├── ensemble_top3_results.csv
    ├── feature_importance_all_models.csv
    ├── shap_feature_importance_xgboost.csv
    └── shap_per_class_importance_xgboost.csv
``` -->

---

## 🎯 **Key Features & Experiments**

### **1. Dataset**
- **Size:** 1,918 balanced samples
- **Target:** CGPA3_Class (3 classes)
  - Class 0 (0.0): CGPA 3.5-4.0 (High)
  - Class 1 (0.5): CGPA 3.0-3.49 (Average)
  - Class 2 (1.0): CGPA < 3.0 (Low)
- **Source:** GAN-augmented synthetic data for balanced classes
- **Normalization:** All features normalized to [0, 1] range

### **2. Models Evaluated**

| Model | Type | Accuracy |
|-------|------|----------|
| **Gradient Boosting** | Ensemble | **63.02%** ⭐ |
| XGBoost | Ensemble | 61.98% |
| CatBoost | Ensemble | 60.42% |
| Random Forest | Ensemble | 60.35% |
| Neural Network (PyTorch) | Deep Learning | 59.53% |
| Logistic Regression | Linear | 59.48% |

### **3. Ensemble Techniques (Top 3 Models)**

9 different ensemble approaches tested:
1. **Soft Voting** - Equal weight probability averaging
2. **Weighted Voting** - Performance-based weight voting
3. **Hard Voting** - Majority vote (best ensemble: 62.76%)
4. **Manual Probability Averaging** - Direct average of probabilities
5. **Weighted Probability Averaging** - Weighted average of probabilities
6. **Dynamic Confidence Weighting** - Confidence-based adaptive weighting
7. **Stacking with Logistic Regression** - Meta-learner approach
8. **Stacking with XGBoost** - XGBoost meta-learner
9. **Blending** - Holdout-based meta-learning

**Finding:** Gradient Boosting (single model) outperformed all ensembles due to limited model diversity.

### **4. Model Interpretability (SHAP Analysis)**

Comprehensive explanation of XGBoost model using SHAP:
- **Global Feature Importance:** Overall model behavior
- **Beeswarm Plots:** Feature impact direction and magnitude
- **Waterfall Plots:** Individual prediction explanations
- **Dependence Plots:** Feature-outcome relationships
- **Per-Class Analysis:** Different drivers for each CGPA class

**Top Predictive Features:**
1. Sleepiness_During_Class (0.0655)
2. Focus_on_Academic_Task (0.0539)
3. Class_Attendance (0.0291)
4. Main_Reason_for_Insufficient_Sleep (0.0272)
5. Rate_Sleep_Quality (0.0266)

---

## 📊 **Model Performance Metrics**

### **Best Model: Gradient Boosting**
```
Accuracy:           63.02%
Macro F1-Score:     62.58%
Balanced Accuracy:  62.63%
ROC-AUC:           0.797
```

### **Per-Class Performance**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| High CGPA | 0.58 | 0.72 | 0.64 |
| Avg CGPA | 0.56 | 0.58 | 0.57 |
| Low CGPA | 0.74 | 0.57 | 0.65 |

---

## 🔍 **Key Findings**

### **Sleep Patterns Matter Most**
- Sleepiness during class is the #1 feature (importance: 0.0655)
- Sleep quality is critical for predicting Low CGPA students
- Students with good sleep are 92% more likely to have High CGPA

### **Academic Focus is Important**
- Focus on tasks is the #2 predictor (importance: 0.0539)
- Lack of focus strongly indicates Low CGPA (95.7% confidence)
- More critical for Low CGPA prediction than High CGPA

### **Attendance Matters**
- Class attendance ranked #3 but less dominant than sleep
- More important for High CGPA students

### **Ensemble Paradox**
- Single strong learner > ensemble of weak learners
- Model diversity is key; all tree-based models led to diminishing returns

---

## 🚀 **How to Run the Notebook**

### **Requirements**
```bash
pip install pandas numpy scikit-learn xgboost catboost torch matplotlib seaborn shap
```

### **Running Steps**
1. Open `06_Balanced_Data_Model_Experiments_CGPA3.ipynb` in Jupyter
2. Run cells sequentially (marked by ## sections)
3. All outputs saved to `images/` and `results/` folders

### **GPU Support (Optional)**
For faster Neural Network training:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # Shows CUDA if available
```

---

## 📈 **Output Files Generated**

### **CSV Results**
- `final_model_comparison.csv` - All 6 models performance
- `ensemble_top3_results.csv` - 9 ensemble methods comparison
- `feature_importance_all_models.csv` - Feature importance from all models
- `shap_feature_importance_xgboost.csv` - SHAP-based feature importance

### **Visualizations**
- Confusion matrices for all models
- Feature importance comparisons
- Ensemble performance charts
- SHAP beeswarm, waterfall, and dependence plots
- Per-class performance heatmaps

---

## 🔧 **Hyperparameter Settings**

### **Gradient Boosting (Best Model)**
```python
n_estimators=300
max_depth=6
learning_rate=0.05
min_samples_split=5
min_samples_leaf=3
subsample=0.8
max_features='sqrt'
```

### **XGBoost**
```python
n_estimators=300
max_depth=6
learning_rate=0.05
min_child_weight=3
subsample=0.8
colsample_bytree=0.8
gamma=1
```

### **Neural Network (PyTorch)**
```
Architecture: [Input(29) → 128 → 64 → 32 → Output(3)]
Activation: ReLU with 0.3 Dropout
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss
Epochs: 100
Batch Size: 32
```

---

## 📝 **Future Improvements**

1. **Model Ensemble with Diverse Algorithms**
   - Mix tree-based with linear/neural models
   - Expected improvement: +2-3% accuracy

2. **Feature Engineering**
   - Create interaction features (sleep_quality × study_time)
   - Temporal features (sleep patterns over time)

3. **Class Imbalance Handling**
   - Class weights in models
   - SMOTE oversampling
   - Threshold tuning

4. **Cross-Validation**
   - Implement k-fold CV (currently single 80-20 split)
   - Nested CV for hyperparameter tuning

5. **Advanced SHAP Analysis**
   - SHAP interaction values
   - Partial dependence plots (PDP)
   - Individual Conditional Expectation (ICE) plots

---

## 👥 **Research Context**

This project investigates how sleep patterns and behavioral factors predict academic performance (CGPA). The study uses:
- **Data Augmentation:** GAN-generated synthetic data for class balancing
- **Multiple ML Paradigms:** Traditional ML, boosting, deep learning
- **Interpretability:** SHAP for understanding model decisions
- **Rigorous Evaluation:** Multiple metrics, confusion matrices, per-class analysis

**Target Audience:** Students, educators, sleep researchers, ML practitioners

---

## 📖 **References & Libraries**

- **XGBoost:** Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
- **SHAP:** Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- **CatBoost:** Dorogush et al. (2018) - "CatBoost: Gradient Boosting for Categorical Data"
- **PyTorch:** Facebook AI Research

---

# License

This repository is distributed under the **Academic Peer Review License (APRL) v1.0**.

The repository is publicly accessible solely for peer review of the accompanying manuscript and is **not an open-source project**.

See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for complete terms.

---

**Last Updated:** 2024  
**Status:** ✅ Complete - All experiments executed successfully
