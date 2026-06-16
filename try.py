import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold  # שונה מ-train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

#Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
#Create dummy data
np.random.seed(42)
n_samples = 500

dummy_data = pd.DataFrame({
    # הוספנו עמודת נבדקים פיקטיבית (25 נבדקים, 20 נקודות זמן לכל אחד) כדי שהשלד ירוץ
    'UserName': np.repeat([f'Subject_{i}' for i in range(1, 26)], 20),
    'KCAL': np.random.normal(2000, 400, n_samples),       #Calories
    'FIBE': np.random.normal(25, 10, n_samples),          #Fiber
    'CARB': np.random.normal(250, 50, n_samples),         #Carbs
    'TFAT': np.random.normal(70, 20, n_samples),          #Total Fat
    'Dietary_Diversity': np.random.uniform(10, 30, n_samples), #Dietary Diversity
    'FME_Score': np.random.normal(500, 150, n_samples)    #FME Score
})

#Create target variable (binary classification)
prob = (dummy_data['FIBE'] * 0.4 + dummy_data['FME_Score'] * 0.1) / 100
dummy_data['High_Diversity_Class'] = np.where(prob + np.random.normal(0, 0.2, n_samples) > 0.5, 1, 0)

print("=== Sample of Dummy Data ===")
print(dummy_data.head())

#Prepare data for modeling
# כאן אנחנו מפרידים גם את עמודת ה-UserName כדי שלא תיכנס למודל כמאפיין
X = dummy_data.drop(['High_Diversity_Class', 'UserName'], axis=1)
y = dummy_data['High_Diversity_Class']
groups = dummy_data['UserName']  # הגדרת הקבוצות לפיהן נחתוך

#Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)  # תוקן: use_label_encoder הוסר
}

# הגדרת מקרוס-וולידציה לפי קבוצות (5 Folds)
gkf = GroupKFold(n_splits=5)

#Train and evaluate models
print("\n=== Model Comparison Results ===")
best_model_name = ""
best_auc = 0
best_model_feat_imp = None  # משתנה לשמירת חשיבות המאפיינים של המודל המנצח

for name, model in models.items():
    # רשימות לשמירת הציונים של כל Fold
    acc_scores = []
    auc_scores = []
    
    # הלולאה רצה על ה-Folds השונים ומוודאת שנבדקים לא מתערבבים
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        #Scale features - מבוצע בתוך הלולאה כדי למנוע זליגת נתונים!
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        #Train
        model.fit(X_train_scaled, y_train)
        
        #Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        #Calculate metrics עבור ה-Fold הנוכחי
        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
    
    # חישוב הממוצע מכל ה-Folds
    mean_acc = np.mean(acc_scores)
    mean_auc = np.mean(auc_scores)
    
    print(f"\n[{name}]")
    print(f"Mean Accuracy: {mean_acc:.3f}")
    print(f"Mean ROC-AUC:  {mean_auc:.3f}")
    
    #Save best model based on AUC
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model_name = name
        # שומרים את חשיבות המאפיינים אם המודל הנוכחי הוא המנצח
        if hasattr(model, 'feature_importances_'):
            best_model_feat_imp = model.feature_importances_

print(f"\n Best Model: {best_model_name} (Mean AUC = {best_auc:.3f})")

if best_model_name in ["Random Forest", "XGBoost"] and best_model_feat_imp is not None:
    importances = best_model_feat_imp
    
elif best_model_name == "Logistic Regression":
    # ב-Logistic Regression יש coefficients במקום feature_importances_
    importances = np.abs(model.coef_[0])

# גרף — אותו קוד לשניהם
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title(f'Feature Importance ({best_model_name})', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Dietary Features', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Plot saved successfully to 'feature_importance.png'")
