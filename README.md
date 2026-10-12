# QSkill AI & ML Internship – Task Submission

**Name:** Isaac Benny
**Domain:** Artificial Intelligence & Machine Learning
**Internship period:** 14 Sep 2026 – 14 Oct 2026
**Tasks completed:** 1. Iris Flower Classification · 3. House Price Prediction

## Project structure
```
QSkill_AI_ML_Internship/
├── task1_iris_classification.py      # Task 1
├── task3_house_price_prediction.py   # Task 3
├── requirements.txt
├── outputs/                          # generated plots
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python task1_iris_classification.py
python task3_house_price_prediction.py
```
Plots are saved to the `outputs/` folder; metrics are printed in the terminal.

---

## Task 1 – Iris Flower Classification
**Goal:** classify iris flowers into Setosa, Versicolor and Virginica from sepal/petal measurements.

**Approach**
1. Loaded the Iris dataset from scikit-learn (150 samples, 4 features, no missing values).
2. Explored it with a pair plot and histograms.
3. Stratified 80/20 train/test split (`random_state=42`).
4. Standardised features with `StandardScaler` (fit on train only).
5. Trained Logistic Regression, K-Nearest Neighbors (k=5) and a Decision Tree (max depth 3).
6. Evaluated with accuracy, macro precision, classification report and confusion matrices.

**Results (30-sample test set)**

| Model | Accuracy | Precision (macro) |
|---|---|---|
| Logistic Regression | 0.933 | 0.933 |
| KNN (k=5) | 0.933 | 0.944 |
| Decision Tree | 0.967 | 0.970 |

Setosa is perfectly separable; the few errors are between Versicolor and Virginica, whose petal measurements overlap. With only 30 test samples, one mistake changes accuracy by ~3%, so the gap between models is small.

Plots: `outputs/iris_pairplot.png`, `outputs/iris_histograms.png`, `outputs/iris_confusion_matrices.png`

---

## Task 3 – House Price Prediction
**Goal:** predict house prices from numeric features using Linear Regression.

**Approach**
1. Loaded the California Housing dataset from scikit-learn (the Boston dataset was removed from scikit-learn over ethical concerns).
2. Explored distributions and the correlation heatmap.
3. Filled any missing values with the column median.
4. 80/20 train/test split, then standardised features (scaler fit on train only to avoid data leakage).
5. Trained `LinearRegression` and evaluated with MSE, RMSE, MAE and R².

**Results (California Housing, 20,640 samples, 8 features, 20% test set)**

| Metric | Value |
|---|---|
| MSE | 0.5559 |
| RMSE | 0.7456 |
| MAE | 0.5332 |
| R² | 0.5758 |

The target (`MedHouseVal`) is in units of $100,000, so the RMSE of 0.7456 means predictions are off by about $74,500 on average. An R² of 0.58 means the model explains roughly 58% of the variance in prices, which is typical for plain Linear Regression on this dataset. Latitude and Longitude have the largest negative coefficients, showing that location strongly drives price. Housing prices depend non-linearly on location, so models such as Random Forest or Gradient Boosting would likely improve the score.

Plots (generated on run): `outputs/house_distributions.png`, `outputs/house_correlation.png`, `outputs/house_results.png`

---

## Skills demonstrated
Data exploration and visualisation · train/test splitting · feature scaling · classification and regression · model evaluation (accuracy, precision, confusion matrix, MSE/RMSE/R²) · avoiding data leakage

**Contact:** isaacbennyv@gmail.com · [GitHub](https://github.com/isaacbenny-max) · [LinkedIn](https://www.linkedin.com/in/isaac-vb)
