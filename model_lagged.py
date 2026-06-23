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
from sklearn.linear_model import Ridge

from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


#Load data
df_fme = pd.read_csv(DATA_DIR / "fme_statistical_dataset_extended.csv", index_col=0)
print(f"Loaded dataset: {df_fme.shape[0]} samples, {df_fme.shape[1]} columns")
print(f"Participants: {df_fme['participant_id'].nunique()}")
print("Sample of Data:")
print(df_fme.head())

#Turn shannon diversity from contiues to binary
median_shannon = df_fme["shannon_diversity"].median()
df_fme["High_Diversity_Class"] = (df_fme["shannon_diversity"] > median_shannon).astype(int)
print(f"\nTarget distribution (High=1 / Low=0):")
print(df_fme["High_Diversity_Class"].value_counts())

#Features
feature_cols_lagged = [
    "fme_score_daily",
    "fme_participant_mean",
    "fme_within_person",
    "fme_lag1",
    "fme_lag2",
    "KCAL",
    "FIBE",
    "CARB",
    "TFAT",
    "PROT",
    "Age",
    "BMI"
]

feature_cols_lagged = [f for f in feature_cols_lagged if f in df_fme.columns]


#Drop rows with missing values in features or target
df_fme_clean = df_fme[feature_cols_lagged + ["High_Diversity_Class", "participant_id"]].dropna()

#Check how many samples are left
print(f"Lagged classification clean dataset: {df_fme_clean.shape[0]} samples")
print(f"Participants after cleaning: {df_fme_clean['participant_id'].nunique()}")

#Prepare data for modeling
X = df_fme_clean[feature_cols_lagged]
print(X.columns.tolist())
y = df_fme_clean['High_Diversity_Class']
groups = df_fme_clean['participant_id']

#Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "SVM": CalibratedClassifierCV(SVC(kernel='rbf'), ensemble=False),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

#Cross validation
gkf = GroupKFold(n_splits=5)

#Train and evaluate models
print("\nModel Comparison Results:")
best_model_name = ""
best_auc = 0
best_model_obj  = None
classification_results = []

for name, model in models.items():
    acc_scores = []
    auc_scores = []
    
    #Make sure subjects are not split between train and test sets
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        #Scale features
        scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols_lagged)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols_lagged)
        
        #Train
        #Clone model to avoid reusing fitted state between folds
        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train)
        
        #Predictions
        y_pred = fold_model.predict(X_test_scaled)
        y_proba = fold_model.predict_proba(X_test_scaled)[:, 1]
        
        #Calculate metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
    
    #Compute mean metrics across folds
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print(f"\n[{name}]")
    print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Mean ROC-AUC:  {mean_auc:.3f} ± {std_auc:.3f}")
    
    classification_results.append({
    "Model": name,
    "Mean Accuracy": mean_acc,
    "Std Accuracy": std_acc,
    "Mean ROC-AUC": mean_auc,
    "Std ROC-AUC": std_auc
})
    
    #Save best model based on AUC
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model_name = name
        best_model_obj  = clone(model)

print(f"\n Best Model: {best_model_name} (Mean AUC = {best_auc:.3f})")

# Save and plot classification model comparison
classification_results_df = pd.DataFrame(classification_results)

classification_results_path = DATA_DIR / "lagged_classification_model_comparison.csv"
classification_results_df.to_csv(classification_results_path, index=False)

print(f"Classification results saved: {classification_results_path}")
print(classification_results_df)

# Plot Accuracy and ROC-AUC comparison
plot_df = classification_results_df.copy()

x = np.arange(len(plot_df["Model"]))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    plot_df["Mean Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x + width / 2,
    plot_df["Mean ROC-AUC"],
    width,
    label="ROC-AUC"
)

plt.axhline(
    y=0.5,
    linestyle="--",
    linewidth=1.5,
    label="Random baseline"
)

plt.xticks(x, plot_df["Model"], rotation=20, ha="right")
plt.ylabel("Score")
min_score = plot_df[["Mean Accuracy", "Mean ROC-AUC"]].min().min()
max_score = plot_df[["Mean Accuracy", "Mean ROC-AUC"]].max().max()

plt.ylim(
    max(0, min_score - 0.05),
    min(1, max_score + 0.08)
)
plt.title("Classification Model Performance")
plt.legend()

for i, row in plot_df.iterrows():
    plt.text(
        i - width / 2,
        row["Mean Accuracy"] + 0.005,
        f"{row['Mean Accuracy']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

    plt.text(
        i + width / 2,
        row["Mean ROC-AUC"] + 0.005,
        f"{row['Mean ROC-AUC']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(DATA_DIR / "lagged_classification_model_performance.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Classification performance plot saved: {DATA_DIR / 'classification_model_performance.png'}")

#Refit best  model on the full clean regression dataset
final_reg_scaler = StandardScaler()
X_scaled = final_reg_scaler.fit_transform(X)
best_model_obj.fit(X_scaled, y)


#Feature importance for best modelv
if best_model_name == "Logistic Regression":
    importances = np.abs(best_model_obj.coef_[0])
    title       = "Feature Importance (Logistic Regression - Absolute Coefficients)"
    xlabel      = "Absolute Coefficient Value"
elif best_model_name == "SVM":
    print("SVM does not support feature importance with rbf kernel - skipping plot")
    importances = None
else:
    importances = best_model_obj.feature_importances_
    title       = f"Feature Importance ({best_model_name})"
    xlabel      = "Importance Score"
 
if importances is not None:
    indices      = np.argsort(importances)[::-1]
    feature_arr  = np.array(feature_cols_lagged)
 
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_arr[indices], hue=feature_arr[indices], palette="viridis", legend=False)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "lagged_feature_importance.png", dpi=150)      #Save before showings
    plt.show()
    print(f"Plot saved: {DATA_DIR / 'lagged_feature_importance.png'}")



#Regression: predict continuous Shannon diversity
print("\n Regression: Predict Shannon Diversity")

#Use the same features, but keep Shannon diversity as a continuous target
df_reg_clean = df_fme[feature_cols_lagged + ["shannon_diversity", "participant_id"]].dropna()

#Prepare regression data
X_reg = df_reg_clean[feature_cols_lagged]
y_reg = df_reg_clean["shannon_diversity"]
groups_reg = df_reg_clean["participant_id"]

print(f"Regression dataset: {df_reg_clean.shape[0]} samples")
print(f"Features used: {feature_cols_lagged}")
print(f"Target mean Shannon diversity: {y_reg.mean():.3f}")
print(f"Target std Shannon diversity:  {y_reg.std():.3f}")

# Define regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
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

# Use grouped cross validation to avoid participant leakage
gkf_reg = GroupKFold(n_splits=5)

#Train and evaluate regression models
print("\nRegression Model Comparison Results")

best_reg_model_name = ""
best_reg_r2 = -np.inf
best_reg_model_template = None

regression_results = []
all_y_true_best = []
all_y_pred_best = []

for name, model in regression_models.items():
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    all_y_true = []
    all_y_pred = []

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
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

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
        all_y_true_best = all_y_true
        all_y_pred_best = all_y_pred

print(f"\nBest Regression Model: {best_reg_model_name} (Mean R² = {best_reg_r2:.3f})")

# Save regression results table
regression_results_df = pd.DataFrame(regression_results)
regression_results_path = DATA_DIR / "lagged_regression_model_comparison.csv"
regression_results_df.to_csv(regression_results_path, index=False)
print(f"lagged_regression_model_comparison: {regression_results_path}")

#Refit best regression model on the full clean regression dataset
final_reg_scaler = StandardScaler()
X_reg_scaled = final_reg_scaler.fit_transform(X_reg)

best_reg_model_obj = clone(best_reg_model_template)
best_reg_model_obj.fit(X_reg_scaled, y_reg)

#Feature importance for best regression model
if best_reg_model_name == "Linear Regression" or best_reg_model_name == "Ridge Regression" :
    reg_importances = np.abs(best_reg_model_obj.coef_)
    reg_title = "Feature Importance ("+best_reg_model_name+" — Absolute Coefficients)"
    reg_xlabel = "Absolute Coefficient Value"
else:
    reg_importances = best_reg_model_obj.feature_importances_
    reg_title = f"Feature Importance ({best_reg_model_name})"
    reg_xlabel = "Importance Score"

reg_indices = np.argsort(reg_importances)[::-1]
feature_arr = np.array(feature_cols_lagged)

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
plt.savefig(DATA_DIR / "lagged_regression_feature_importance.png", dpi=150)
plt.show()
print(f"Regression feature importance plot saved: {DATA_DIR / 'lagged_regression_feature_importance.png'}")


all_y_true = np.array(all_y_true_best)
all_y_pred = np.array(all_y_pred_best)

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
plt.savefig(DATA_DIR / "lagged_regression_predicted_vs_actual.png", dpi=150)
plt.show()
print(f"Prediction plot saved: {DATA_DIR / 'lagged_regression_predicted_vs_actual.png'}")

#Plot residuals
residuals = all_y_true - all_y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=all_y_pred, y=residuals)
plt.axhline(0, linestyle="--")
plt.title(f"Residual Plot ({best_reg_model_name})", fontsize=14)
plt.xlabel("Predicted Shannon Diversity", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "lagged_regression_residuals.png", dpi=150)
plt.show()
print(f"Residual plot saved: {DATA_DIR / 'lagged_regression_residuals.png'}")

#  Try model distance from avarge shanon instead of predict shanon
# Calculate avg shanon score
df_fme['shannon_avg'] = df_fme.groupby('participant_id')['shannon_diversity'].transform('mean')
# Calculate sample shanon deviation
df_fme['shannon_deviation'] = df_fme['shannon_diversity'] - df_fme['shannon_avg']

#Regression: predict continuous distance from avg Shannon diversity
print("\n Regression: predict continuous distance from avg Shannon diversity")

#Use the same features, but keep Shannon diversity as a continuous target
df_reg_dev_clean = df_fme[feature_cols_lagged + ["shannon_deviation", "participant_id"]].dropna()

#Prepare regression data
X_dev_reg = df_reg_dev_clean[feature_cols_lagged]
y_dev_reg = df_reg_dev_clean["shannon_deviation"]
groups_dev_reg = df_reg_dev_clean["participant_id"]

print(f"Regression dataset: {df_reg_dev_clean.shape[0]} samples")
print(f"Features used: {feature_cols_lagged}")
print(f"Target mean Shannon deviation diversity: {y_dev_reg.mean():.3f}")
print(f"Target std Shannon deviation diversity:  {y_dev_reg.std():.3f}")

#Define regression models
regression_dev_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
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
print("\nRegression Model for deviation- comparison Results")

best_dev_reg_model_name = ""
best_dev_reg_r2 = -np.inf
best_dev_reg_model_template = None

regression_dev_results = []
all_dev_y_true_best = []
all_dev_y_pred_best = []

for name, model in regression_dev_models.items():
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    all_y_true = []
    all_y_pred = []

    #Make sure subjects are not split between train and test sets
    for train_idx, test_idx in gkf_reg.split(X_dev_reg, y_dev_reg, groups=groups_dev_reg):
        X_train, X_test = X_dev_reg.iloc[train_idx], X_dev_reg.iloc[test_idx]
        y_train, y_test = y_dev_reg.iloc[train_idx], y_dev_reg.iloc[test_idx]

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
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    #Compute mean metrics across folds
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)

    regression_dev_results.append({
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
    if mean_r2 > best_dev_reg_r2:
        best_dev_reg_r2 = mean_r2
        best_dev_reg_model_name = name
        best_dev_reg_model_template = model
        all_dev_y_true_best = all_y_true
        all_dev_y_pred_best = all_y_pred

print(f"\nBest Regression Model: {best_dev_reg_model_name} (Mean R² = {best_dev_reg_r2:.3f})")

#Save regression  deviation from avg results table
regression_dev_results_df = pd.DataFrame(regression_dev_results)
regression_dev_results_path = DATA_DIR / "lagged_regression_dev_model_comparison.csv"
regression_dev_results_df.to_csv(regression_dev_results_path, index=False)
print(f"Regression results saved: {regression_dev_results_path}")

#Refit best regression model on the full clean regression dataset
final_dev_reg_scaler = StandardScaler()
X_dev_reg_scaled = final_dev_reg_scaler.fit_transform(X_dev_reg)

best_dev_reg_model_obj = clone(best_dev_reg_model_template)
best_dev_reg_model_obj.fit(X_dev_reg_scaled, y_dev_reg)

#Feature importance for best regression model
if best_dev_reg_model_name == "Linear Regression" or best_dev_reg_model_name == "Ridge Regression" :
    reg_dev_importances = np.abs(best_dev_reg_model_obj.coef_)
    reg_dev_title = "Feature Importance ("+best_dev_reg_model_name+" - Absolute Coefficients)"
    reg_dev_xlabel = "Absolute Coefficient Value"
else:
    reg_dev_importances = best_dev_reg_model_obj.feature_importances_
    reg_dev_title = f"Feature Importance ({best_dev_reg_model_name})"
    reg_dev_xlabel = "Importance Score"

reg_dev_indices = np.argsort(reg_dev_importances)[::-1]
feature_dev_arr = np.array(feature_cols_lagged)

# Bar plot of distribution of Shannon Deviation from Personal Average
plt.figure(figsize=(8, 5))
sns.histplot(df_reg_dev_clean["shannon_deviation"], bins=30, kde=True)
plt.axvline(0, linestyle="--", color="red")
plt.title("Distribution of Shannon Deviation from Personal Average", fontsize=14)
plt.xlabel("Shannon Deviation", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "lagged_dev_shannon_deviation_distribution.png", dpi=150)
plt.show()

# Bar plot of feature importance
plt.figure(figsize=(10, 6))
sns.barplot(
    x=reg_dev_importances[reg_dev_indices],
    y=feature_dev_arr[reg_dev_indices],
    hue=feature_dev_arr[reg_dev_indices],
    palette="viridis",
    legend=False
)
plt.title(reg_dev_title, fontsize=14)
plt.xlabel(reg_dev_xlabel, fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "lagged_dev_regression_feature_importance.png", dpi=150)
plt.show()
print(f"Regression feature importance plot saved: {DATA_DIR / 'lagged_dev_regression_feature_importance.png'}")


all_dev_y_true = np.array(all_dev_y_true_best)
all_dev_y_pred = np.array(all_dev_y_pred_best)

#Plot predicted vs actual Shannon diversity
plt.figure(figsize=(7, 7))
sns.scatterplot(x=all_dev_y_true, y=all_dev_y_pred)
plt.plot(
    [all_dev_y_true.min(), all_dev_y_true.max()],
    [all_dev_y_true.min(), all_dev_y_true.max()],
    linestyle="--"
)
plt.title(f"Predicted vs Actual Shannon Deviation  ({best_dev_reg_model_name})", fontsize=14)
plt.xlabel("Actual Shannon Deviation", fontsize=12)
plt.ylabel("Predicted Shannon Deviation", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "lagged_dev_regression_predicted_vs_actual.png", dpi=150)
plt.show()
print(f"Prediction plot saved: {DATA_DIR / 'lagged_dev_regression_predicted_vs_actual.png'}")

#Plot residuals
dev_residuals = all_dev_y_true - all_dev_y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=all_dev_y_pred, y=dev_residuals)
plt.axhline(0, linestyle="--")
plt.title(f"Residual Plot ({best_dev_reg_model_name})", fontsize=14)
plt.xlabel("Predicted Shannon Deviation", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "lagged_dev_regression_residuals.png", dpi=150)
plt.show()
print(f"Residual plot saved: {DATA_DIR / 'lagged_dev_regression_residuals.png'}")