import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

#Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Regression imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

from config import DATA_DIR

#Load data
df_fme = pd.read_csv(DATA_DIR / "fme_statistical_dataset.csv", index_col=0)
print(f"Loaded dataset: {df_fme.shape[0]} samples, {df_fme.shape[1]} columns")
print(f"Participants: {df_fme['participant_id'].nunique()}")



print("=== Sample of  Data ===")
print(df_fme.head())

#Turn shannon diversity from contiues to binary
median_shannon = df_fme["shannon_diversity"].median()
df_fme["High_Diversity_Class"] = (df_fme["shannon_diversity"] > median_shannon).astype(int)
print(f"\nTarget distribution (High=1 / Low=0):")
print(df_fme["High_Diversity_Class"].value_counts())

#Features
feature_cols = ["fme_score_daily", "KCAL", "FIBE", "CARB", "TFAT", "Age", "BMI"]
feature_cols = [f for f in feature_cols if f in df_fme.columns]

#Drop rows with missing values in features or target
df_fme_clean = df_fme[feature_cols + ["High_Diversity_Class", "participant_id"]].dropna()

#Prepare data for modeling
X = df_fme_clean[feature_cols]
print(X.columns.tolist())
y = df_fme_clean['High_Diversity_Class']
groups = df_fme_clean['participant_id']

#Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

#Cross validation
gkf = GroupKFold(n_splits=5)

#Train and evaluate models
print("\n=== Model Comparison Results ===")
best_model_name = ""
best_auc = 0
best_model_feat_obj = None

for name, model in models.items():
    acc_scores = []
    auc_scores = []
    
    #Make sure subjects are not split between train and test sets
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        #Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        #Train
        model.fit(X_train_scaled, y_train)
        
        #Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        #Calculate metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
    
    #Compute mean metrics across folds
    mean_acc = np.mean(acc_scores)
    mean_auc = np.mean(auc_scores)
    
    print(f"\n[{name}]")
    print(f"Mean Accuracy: {mean_acc:.3f}")
    print(f"Mean ROC-AUC:  {mean_auc:.3f}")
    
    #Save best model based on AUC
    if mean_auc > best_auc:
        best_auc       = mean_auc
        best_model_name = name
        best_model_obj  = model

print(f"\n Best Model: {best_model_name} (Mean AUC = {best_auc:.3f})")

#Feature importance for best modelv
if best_model_name == "Logistic Regression":
    importances = np.abs(best_model_obj.coef_[0])
    title       = "Feature Importance (Logistic Regression — Absolute Coefficients)"
    xlabel      = "Absolute Coefficient Value"
else:
    importances = best_model_obj.feature_importances_
    title       = f"Feature Importance ({best_model_name})"
    xlabel      = "Importance Score"
 
indices      = np.argsort(importances)[::-1]
feature_arr  = np.array(feature_cols)
 
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_arr[indices], hue=feature_arr[indices], palette="viridis", legend=False)
plt.title(title, fontsize=14)
plt.xlabel(xlabel, fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "feature_importance.png", dpi=150)      #Save before showings
plt.show()
print(f"Plot saved: {DATA_DIR / 'feature_importance.png'}")

#Regression: predict continuous Shannon diversity
print("\n Regression: Predict Shannon Diversity")

#Use the same features, but keep Shannon diversity as a continuous target
df_reg_clean = df_fme[feature_cols + ["shannon_diversity", "participant_id"]].dropna()

#Prepare regression data
X_reg = df_reg_clean[feature_cols]
y_reg = df_reg_clean["shannon_diversity"]
groups_reg = df_reg_clean["participant_id"]

print(f"Regression dataset: {df_reg_clean.shape[0]} samples")
print(f"Features used: {feature_cols}")
print(f"Target mean Shannon diversity: {y_reg.mean():.3f}")
print(f"Target std Shannon diversity:  {y_reg.std():.3f}")

#Define regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost Regressor": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        random_state=42
    )
}

#Use grouped cross validation to avoid participant leakage
gkf_reg = GroupKFold(n_splits=5)

#Train and evaluate regression models
print("\nRegression Model Comparison Results")

best_reg_model_name = ""
best_reg_r2 = -np.inf
best_reg_model_template = None

regression_results = []

for name, model in regression_models.items():
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    #Make sure subjects are not split between train and test sets
    for train_idx, test_idx in gkf_reg.split(X_reg, y_reg, groups=groups_reg):
        X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

        #Scale features using train data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Clone model to avoid reusing fitted state between folds
        fold_model = clone(model)

        #Train
        fold_model.fit(X_train_scaled, y_train)

        #Predict continuous Shannon diversity
        y_pred = fold_model.predict(X_test_scaled)

        #Calculate regression metrics
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    #Compute mean metrics across folds
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)

    regression_results.append({
        "Model": name,
        "Mean R2": mean_r2,
        "Std R2": std_r2,
        "Mean RMSE": mean_rmse,
        "Mean MAE": mean_mae
    })

    print(f"\n[{name}]")
    print(f"Mean R^2:    {mean_r2:.3f} ± {std_r2:.3f}")
    print(f"Mean RMSE:  {mean_rmse:.3f}")
    print(f"Mean MAE:   {mean_mae:.3f}")

    #Save best regression model based on R2
    if mean_r2 > best_reg_r2:
        best_reg_r2 = mean_r2
        best_reg_model_name = name
        best_reg_model_template = model

print(f"\nBest Regression Model: {best_reg_model_name} (Mean R² = {best_reg_r2:.3f})")

#Save regression results table
regression_results_df = pd.DataFrame(regression_results)
regression_results_path = DATA_DIR / "regression_model_comparison.csv"
regression_results_df.to_csv(regression_results_path, index=False)
print(f"Regression results saved: {regression_results_path}")

#Refit best regression model on the full clean regression dataset
final_reg_scaler = StandardScaler()
X_reg_scaled = final_reg_scaler.fit_transform(X_reg)

best_reg_model_obj = clone(best_reg_model_template)
best_reg_model_obj.fit(X_reg_scaled, y_reg)

#Feature importance for best regression model
if best_reg_model_name == "Linear Regression":
    reg_importances = np.abs(best_reg_model_obj.coef_)
    reg_title = "Feature Importance (Linear Regression — Absolute Coefficients)"
    reg_xlabel = "Absolute Coefficient Value"
else:
    reg_importances = best_reg_model_obj.feature_importances_
    reg_title = f"Feature Importance ({best_reg_model_name})"
    reg_xlabel = "Importance Score"

reg_indices = np.argsort(reg_importances)[::-1]
feature_arr = np.array(feature_cols)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=reg_importances[reg_indices],
    y=feature_arr[reg_indices],
    hue=feature_arr[reg_indices],
    palette="viridis",
    legend=False
)
plt.title(reg_title, fontsize=14)
plt.xlabel(reg_xlabel, fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "regression_feature_importance.png", dpi=150)
plt.show()
print(f"Regression feature importance plot saved: {DATA_DIR / 'regression_feature_importance.png'}")

#Create cross validated predictions for visualization
all_y_true = []
all_y_pred = []

for train_idx, test_idx in gkf_reg.split(X_reg, y_reg, groups=groups_reg):
    X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
    y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    #Scale features using train data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Train fresh model for this fold
    fold_model = clone(best_reg_model_template)
    fold_model.fit(X_train_scaled, y_train)

    #Predict
    y_pred = fold_model.predict(X_test_scaled)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

#Plot predicted vs actual Shannon diversity
plt.figure(figsize=(7, 7))
sns.scatterplot(x=all_y_true, y=all_y_pred)
plt.plot(
    [all_y_true.min(), all_y_true.max()],
    [all_y_true.min(), all_y_true.max()],
    linestyle="--"
)
plt.title(f"Predicted vs Actual Shannon Diversity ({best_reg_model_name})", fontsize=14)
plt.xlabel("Actual Shannon Diversity", fontsize=12)
plt.ylabel("Predicted Shannon Diversity", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "regression_predicted_vs_actual.png", dpi=150)
plt.show()
print(f"Prediction plot saved: {DATA_DIR / 'regression_predicted_vs_actual.png'}")

#Plot residuals
residuals = all_y_true - all_y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=all_y_pred, y=residuals)
plt.axhline(0, linestyle="--")
plt.title(f"Residual Plot ({best_reg_model_name})", fontsize=14)
plt.xlabel("Predicted Shannon Diversity", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "regression_residuals.png", dpi=150)
plt.show()
print(f"Residual plot saved: {DATA_DIR / 'regression_residuals.png'}")