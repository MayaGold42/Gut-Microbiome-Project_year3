import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import seaborn as sns

# Part 1 - Import data
df_merged = pd.read_csv("fme_statistical_dataset.csv", index_col=0)
df_merged = df_merged.sort_values(['participant_id', 'study_day'])
# print("df_merged")
# print(df_merged.head())
micro_clr = pd.read_csv("data\\microbiome_clr.csv", index_col=0).T
# print("micro_clr")
# print(micro_clr.head())
# Take only relavent coulmns
#df_merged=df_merged[['participant_id', 'study_day', 'fme_score_daily', 'shannon_diversity','participant_shannon_cv', 'participant_shannon_mean','participant_n_samples']]

# Part 2 - Calulate microbiom features
#Claculate shannon deviation from shannon mean for each  participant
df_merged['shannon_deviation'] = df_merged['shannon_diversity'] - df_merged['participant_shannon_mean']
#Claculate  shannon stability  for each  participant
df_merged['shannon_stability'] = df_merged.groupby('participant_id')['shannon_diversity'].diff().abs()

# Join samples df with clr data
df_clr_joined = df_merged[['participant_id', 'study_day']].join(micro_clr)
# print("df_clr_joined")
# print(df_clr_joined.head())


# Calculate Aitchison distance between consecutive days for each participant
stability_rows = []
# Iterate on each participant
for pid, group in df_clr_joined.groupby('participant_id'):
    group = group.sort_values('study_day')
    clr_vals = group.drop(columns=['participant_id', 'study_day']).values
    # Iterate on days to compare clr values
    for i in range(len(clr_vals) - 1):
        # Claculate Aitchison distance 
        euc_dist = np.sqrt(np.sum((clr_vals[i] - clr_vals[i+1])**2))
        stability_rows.append({
            'fecal_sample_id': group.index[i+1],
            'aitchison_day_dist': euc_dist
        })
# print("stability_rows")
# print(stability_rows)

# Aitchison list to df so we could join it with samples df
df_aitchison = pd.DataFrame(stability_rows).set_index('fecal_sample_id')
# print("df_aitchison")
# print(df_aitchison.head())

# Add the aitchison to the samples df
df_merged = df_merged.join(df_aitchison)

# Calculate stability baesd on aitchison
participant_stability = (
    df_merged.groupby('participant_id')['aitchison_day_dist']
    .median()
    .apply(lambda x: 1 / x)
    .rename('aitchison_stability')
)
df_merged = df_merged.join(participant_stability, on='participant_id')
# print('saitchison_stability' )
# print(df_merged['aitchison_stability'] )

# print(df_merged['aitchison_day_dist'].isna().sum())
# print(df_merged['aitchison_day_dist'].head(20))

# Create fme fields that take into account previous days fme
df_merged['fme_lag1'] = df_merged.groupby('participant_id')['fme_score_daily'].shift(1)
df_merged['fme_lag2'] = df_merged.groupby('participant_id')['fme_score_daily'].shift(2)
df_merged['fme_lag3'] = df_merged.groupby('participant_id')['fme_score_daily'].shift(3)
df_merged['fme_weighted'] = (0.5 * df_merged['fme_lag1'] +0.3 * df_merged['fme_lag2'] +0.2 * df_merged['fme_lag3'])

print(df_merged.head(10))

# Part 2 - check statistics
print("Part 2 - Calulate microbiom features")
fme_versions = {'same day': 'fme_score_daily','weighted lag': 'fme_weighted'}
all_results ={}

# Check  fme vs Stability per sample
print("Check  fme vs Stability per sample")
# Check both versions of fme
for lag_label, fme_col in fme_versions.items():
    aitchison_rows = []
    # Iterate on each participant
    for subject, group in df_merged.groupby('participant_id'):
        group = group.sort_values('study_day')


        # Check spearmanr btween  fme and aitchison dist     
        valid = group[[fme_col, 'aitchison_day_dist']].dropna()
        if len(valid) < 4:
            continue
        r, p = stats.spearmanr(valid[fme_col], valid['aitchison_day_dist']) 

        aitchison_rows.append({
            'participant_id': subject,
            'n_days': len(group),
            'spearman_r': r,
            'p_value': p,
            'significant': p < 0.05
        })
    all_results[f'Aitchison daily ({lag_label})'] = pd.DataFrame(aitchison_rows)

# Check  fme vs Stability per participant
print("Check  fme vs Stability per participant")
# Check both versions of fme
participant_fme = df_merged.groupby('participant_id')[['fme_score_daily', 'fme_weighted']].mean()
participant_df = participant_fme.join(df_merged.groupby('participant_id')['aitchison_stability'].first())
# print("participant_df")
# print(participant_df.head)
# Check spearmanr btween mean fme and aitchison stability
for lag_label, fme_col in fme_versions.items():
    #valid = participant_df[[fme_col, 'aitchison_stability']].dropna()
    r, p = stats.spearmanr(participant_df[fme_col], participant_df['aitchison_stability'])
    #all_results[f'Aitchison stability ({lag_label})'] = pd.DataFrame(aitchison_rows)
    results = [{'n_days': len(participant_df),
            'spearman_r': r,
            'p_value': p,
            'significant': p < 0.05}]
    #print(f'{lag_label}: r={r:.3f}, p={p:.3f}, n={len(participant_df)}')
    all_results[f'Aitchison stability ({lag_label})'] = pd.DataFrame(results)

# Check results of fme vs Stability
print("Check results of fme vs Stability")
# Print results
print("Print results")
for lag_type,results in all_results.items():
    print(f"lag type {lag_type}")
    print(results)
# Find stats on results
print("Find stats on results")
for outcome_name, df_res in all_results.items():
    valid_r = df_res['spearman_r'].dropna()
    if len(valid_r) > 1:
        t_stat, p_group = stats.ttest_1samp(valid_r, 0)
        print(f"  One-sample t-test: t={t_stat:.3f}, p={p_group:.3f}")
    print(f"\n{outcome_name} (n={len(valid_r)} subjects):")
    print(f"  Positive: {(valid_r > 0).sum()} | Negative: {(valid_r < 0).sum()}")
    print(f"  Significant (p<0.05): {df_res['significant'].sum()}")
    print(f"  Mean r: {valid_r.mean():.3f} | Median r: {valid_r.median():.3f}")

# Personal FME vs Nutritional Features
print("Personal FME vs Nutritional Features")
nutritional_features = ['FIBE', 'KCAL', 'CARB', 'TFAT']
print(" Within-subject: FME vs Nutritional Features")

# Check both versions of fme
for lag_label, fme_col in fme_versions.items():
    print(lag_label)
    feature_rows = []
    # Iterate on each feature
    for feature in nutritional_features:
        feature_rows = []
        # Iterate on each subject
        for subject, group in df_merged.groupby('participant_id'):
            valid = group[[fme_col, feature]].dropna()
            if len(valid) < 4:
                continue
            r, p = stats.spearmanr(valid[fme_col], valid[feature])
            if np.isnan(r):
                continue
            feature_rows.append({
                'participant_id': subject,
                'spearman_r': r,
                'p_value': p,
                'significant': p < 0.05
        })
        #Check stats on results
        df_feature = pd.DataFrame(feature_rows)
        valid_r = df_feature['spearman_r'].dropna()
        t_stat, p_group = stats.ttest_1samp(valid_r, 0)

        print("Print results")
        print(f"\n{feature}:")
        print(f"  Positive: {(valid_r > 0).sum()} | Negative: {(valid_r < 0).sum()}")
        print(f"  Mean r: {valid_r.mean():.3f} | Median r: {valid_r.median():.3f}")
        print(f"  Significant (p<0.05): {df_feature['significant'].sum()}")
        print(f"  t-test: t={t_stat:.3f}, p={p_group:.3f}")




# Part 3 - graphs
# Create graph
# Histogrma graphs
for outcome_name, df_res in all_results.items():
    valid_r = df_res['spearman_r'].dropna()
    if len(valid_r) <= 1:
        continue
    t_stat, p_group = stats.ttest_1samp(valid_r, 0)
    plt.figure(figsize=(8, 4))
    plt.hist(valid_r, bins=10, color='steelblue', edgecolor='white')
    plt.axvline(0, color='red', linestyle='--', label='r=0')
    plt.axvline(valid_r.mean(), color='orange', linestyle='--', label=f'mean={valid_r.mean():.3f}')
    plt.title(f'{outcome_name}\nt={t_stat:.3f}, p={p_group:.3f}')
    plt.xlabel('Spearman r')
    plt.ylabel('Number of subjects')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'hist_{outcome_name}.png', dpi=150)
    plt.show()

# Bar graphs
for outcome_name, df_res in all_results.items():
    valid_df = df_res.dropna(subset=['spearman_r'])
    if len(valid_df) <= 1:
        continue
    df_sorted = valid_df.sort_values('spearman_r')
    colors = ['tomato' if s else 'steelblue' for s in df_sorted['significant']]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(df_sorted)), df_sorted['spearman_r'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.yticks(range(len(df_sorted)), df_sorted['participant_id'], fontsize=8)
    plt.xlabel('Spearman r')
    plt.title(f'{outcome_name}\nPer-Subject (red = p<0.05)')
    plt.tight_layout()
    plt.savefig(f'bar_{outcome_name}.png', dpi=150)
    plt.show()

# Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('FME vs Aitchison Stability (per participant)', fontsize=13)
for ax, (lag_label, fme_col) in zip(axes, fme_versions.items()):
    valid = participant_df[[fme_col, 'aitchison_stability']].dropna()
    r, p = stats.spearmanr(valid[fme_col], valid['aitchison_stability'])
    
    ax.scatter(valid[fme_col], valid['aitchison_stability'], color='steelblue', alpha=0.7)
    ax.set_xlabel(f'Mean FME ({lag_label})')
    ax.set_ylabel('Aitchison Stability')
    ax.set_title(f'{lag_label}\nr={r:.3f}, p={p:.3f}')
    plt.tight_layout()
    plt.savefig('scatter_fme_stability.png', dpi=150)
    plt.show()

# Diff in fme graphs
valid = df_merged[['fme_score_daily', 'fme_weighted']].dropna()

plt.figure(figsize=(7, 7))
plt.scatter(valid['fme_score_daily'], valid['fme_weighted'], 
            color='steelblue', alpha=0.5, s=20)
plt.plot([valid['fme_score_daily'].min(), valid['fme_score_daily'].max()],
         [valid['fme_score_daily'].min(), valid['fme_score_daily'].max()],
         color='red', linestyle='--', label='y=x')
plt.xlabel('FME same day')
plt.ylabel('FME weighted lag')
plt.title('Same-day FME vs Weighted lag FME')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_fme_comparison.png', dpi=150)
plt.show()

# Diff in fme graphs in r
daily_r = all_results['Aitchison daily (same day)'].set_index('participant_id')['spearman_r']
weighted_r = all_results['Aitchison daily (weighted lag)'].set_index('participant_id')['spearman_r']

compare = pd.DataFrame({'same_day': daily_r, 'weighted': weighted_r}).dropna()

plt.figure(figsize=(6, 6))
plt.scatter(compare['same_day'], compare['weighted'], color='steelblue', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.plot([-1, 1], [-1, 1], color='red', linestyle='--', label='y=x')
plt.xlabel('Spearman r (same day FME)')
plt.ylabel('Spearman r (weighted lag FME)')
plt.title('Per-subject correlation: same day vs weighted lag')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_r_comparison.png', dpi=150)
plt.show()

# HeatMap
features_corr = ['fme_score_daily', 'fme_weighted', 'FIBE', 'KCAL', 'CARB', 'TFAT', 'Age', 'BMI', 'aitchison_day_dist', 'shannon_diversity']

corr_matrix = df_merged[features_corr].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# Models
print("****Models****")

# General models
print("General models")
# All features
print(" All features")
features = ['fme_score_daily', 'fme_weighted', 'FIBE', 'KCAL', 'CARB', 'TFAT', 'Age', 'BMI']
target = 'aitchison_day_dist'
# Prepare the data
df_model = df_merged[features + [target, 'participant_id']].dropna()
X = df_model[features]
y = df_model[target]
groups = df_model['participant_id']

# GroupKFold  so that the same participant is not in train and test
gkf = GroupKFold(n_splits=5)
# Try regression models
models = {
    'Linear Regression': 
    Pipeline([('scaler', StandardScaler()),('model', LinearRegression())
    ]),
    'Ridge': 
    Pipeline([('scaler', StandardScaler()),('model', Ridge())
    ]),
    'Random Forest':
    Pipeline([('scaler', StandardScaler()),('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}

print("Regression: Predict Aitchison Day Distance")
print(f"Dataset: {len(df_model)} samples, {len(df_model['participant_id'].unique())} participants")

#Iterate on models
for model_name, pipeline in models.items():
    #Using cross_val_score function to activate them
    # Check r2 of the models
    r2_scores = cross_val_score(pipeline, X, y, groups=groups, cv=gkf, scoring='r2')
    # Check error of models
    rmse_scores = cross_val_score(pipeline, X, y, groups=groups, cv=gkf, scoring='neg_root_mean_squared_error')
    
    print(f"\n[{model_name}]")
    print(f"  Mean R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
    print(f"  Mean RMSE: {(-rmse_scores).mean():.3f}")

# Only fme model
print(" Only FME")
features_fme_only = ['fme_score_daily']
# Prepare the data
df_model_simple = df_merged[features_fme_only + [target, 'participant_id']].dropna()
X_simple = df_model_simple[features_fme_only]
y_simple = df_model_simple[target]
groups_simple = df_model_simple['participant_id']
models_simple = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
}

print("Regression: FME only to Aitchison Day Distance ")
#Iterate on models
for model_name, pipeline in models_simple.items():
    # Check r2 of the models
    r2_scores = cross_val_score(pipeline, X_simple, y_simple, groups=groups_simple, cv=gkf, scoring='r2')
    # Check error of models
    rmse_scores = cross_val_score(pipeline, X_simple, y_simple, groups=groups_simple, cv=gkf, scoring='neg_root_mean_squared_error')
    print(f"\n[{model_name}]")
    print(f"  Mean R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
    print(f"  Mean RMSE: {(-rmse_scores).mean():.3f}")


# Persenolized models
print("Persenolized models")
# All features
print(" All features")
features_personal = ['fme_score_daily', 'FIBE', 'KCAL', 'CARB', 'TFAT']
target = 'aitchison_day_dist'
personal_coefs = []
# Iterate on subjects
for subject, group in df_merged.groupby('participant_id'):
    valid = group[features_personal + [target]].dropna()
    if len(valid) < 5:
        continue
    X_sub = valid[features_personal]
    y_sub = valid[target]
    # activate model
    model = LinearRegression()
    model.fit(X_sub, y_sub)

    personal_coefs.append({
        'participant_id': subject,
        'fme_score_daily':  model.coef_[0],
        'FIBE':  model.coef_[1],
        'KCAL':  model.coef_[2],
        'CARB':  model.coef_[3],
        'TFAT':  model.coef_[3],
    })

df_coefs = pd.DataFrame(personal_coefs)
# Check stats on coefs
for feature in features_personal:
    print(feature)
    t_stat, p_val = stats.ttest_1samp(df_coefs[feature], 0)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df_coefs[feature])
    print(f"Mean coefficient: {df_coefs[feature].mean():.4f}")
    print(f"Positive: {(df_coefs[feature] > 0).sum()} | Negative: {(df_coefs[feature] < 0).sum()}")
    print(f"t-test: t={t_stat:.3f}, p={p_val:.3f}")
    print(f"Wilcoxon: stat={wilcoxon_stat:.3f}, p={wilcoxon_p:.3f}")

# Only fme
print(" Only fme")
features_fme_only = ['fme_score_daily']
target = 'aitchison_day_dist'
personal_coefs = []
# Prepare the data
df_model_simple = df_merged[features_fme_only + [target, 'participant_id']].dropna()
X_simple = df_model_simple[features_fme_only]
y_simple = df_model_simple[target]
# Iterate on subjects
for subject, group in df_model_simple.groupby('participant_id'):
    valid = group[features_fme_only + [target]].dropna()
    if len(valid) < 5:
        continue
    X_sub = valid[features_fme_only]
    y_sub = valid[target]
    # activate model
    model = LinearRegression()
    model.fit(X_sub, y_sub)

    personal_coefs.append({
        'participant_id': subject,
        'fme_score_daily':  model.coef_[0],
    })

df_coefs = pd.DataFrame(personal_coefs)
# Check stats on coefs
print("fme_score_daily")
t_stat, p_val = stats.ttest_1samp(df_coefs["fme_score_daily"], 0)
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df_coefs["fme_score_daily"])
print(f"Mean coefficient: {df_coefs["fme_score_daily"].mean():.4f}")
print(f"Positive: {(df_coefs["fme_score_daily"] > 0).sum()} | Negative: {(df_coefs["fme_score_daily"] < 0).sum()}")
print(f"t-test: t={t_stat:.3f}, p={p_val:.3f}")
print(f"Wilcoxon: stat={wilcoxon_stat:.3f}, p={wilcoxon_p:.3f}")

