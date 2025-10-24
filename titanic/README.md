# Titanic ML Project

## 🎯 Project Goal

Port the Kaggle Titanic Tutorial from Python (pandas + scikit-learn) to Rust using:

- **Polars** as a pandas replacement for data manipulation
- **Linfa** as a scikit-learn replacement for machine learning

## 🔮 Future Improvements

- [ ] Fix feature alignment between train/test (handle one-hot encoding differences)
- [ ] Add feature subsetting at each split (true RF behavior)
- [ ] Implement out-of-bag (OOB) error estimation
- [ ] Add feature importance calculation
- [ ] Cross-validation support
- [ ] Handle more features (Age, Fare, Cabin, etc.)
- [ ] Better missing value imputation
- [ ] Hyperparameter tuning
- [ ] Add more ML models (Gradient Boosting, etc.)

## Iterations on Predictions

1. `0.1.0` submission: 0.76555
2. `0.2.0` submission: 0.7837
   - Added `test_train_split`
