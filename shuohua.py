import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 样本量
n_samples = 500

# 1. 生成基本信息数据
data = pd.DataFrame()
data["Q1"] = np.ones(n_samples, dtype=int)
data["Q2"] = np.random.choice([1, 2], size=n_samples)
data["Q3"] = np.random.choice([1, 2, 3], size=n_samples)
data["Q4"] = np.random.choice([1, 2, 3], size=n_samples)

# 排序题
for i in range(5):
    data[f"rank_{i+1}"] = [
        np.random.permutation([1, 2, 3, 4, 5])[i] for _ in range(n_samples)
    ]

# 单选题
data["Q6"] = np.random.choice([1, 2, 3, 4, 5], size=n_samples)

# 生成标准化的潜在变量（均值0，标准差1），使交互效应更可控
A_latent = np.random.normal(0, 1, n_samples)  # 师生关系（已标准化）
C_latent = np.random.normal(0, 1, n_samples)  # 自我效能（已标准化）

# B = 基准值 - A的主效应 - C的主效应 - A*C的交互效应 + 误差
# 交互项为负，确保A和C同时增大时，B显著减小
B_latent = (
    3
    - 0.5 * A_latent
    - 0.5 * C_latent
    - 0.6 * (A_latent * C_latent)
    + np.random.normal(0, 0.3, n_samples)
)
B_latent = np.clip(B_latent, 1, 5)  # 限制在1-5分制

# 3. 生成量表观测题
# A量表：q7-q26（20题）
for i in range(20):
    data[f"q{i+7}"] = 3 + A_latent + np.random.normal(0, 0.3, n_samples)
    data[f"q{i+7}"] = np.clip(data[f"q{i+7}"], 1, 5).round(0).astype(int)

# B量表：q27-q46（20题）
for i in range(20):
    data[f"q{i+27}"] = B_latent + np.random.normal(0, 0.3, n_samples)
    data[f"q{i+27}"] = np.clip(data[f"q{i+27}"], 1, 5).round(0).astype(int)

# C量表：q47-q63（17题）
for i in range(17):
    data[f"q{i+47}"] = 3 + C_latent + np.random.normal(0, 0.3, n_samples)
    data[f"q{i+47}"] = np.clip(data[f"q{i+47}"], 1, 5).round(0).astype(int)


# 4. 信度分析
def cronbach_alpha(items):
    items = np.asarray(items)
    k = items.shape[1]
    item_var = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)


A_items = data.iloc[:, 8:28]
B_items = data.iloc[:, 28:48]
C_items = data.iloc[:, 48:65]

alpha_A = cronbach_alpha(A_items)
alpha_B = cronbach_alpha(B_items)
alpha_C = cronbach_alpha(C_items)

# 5. 计算总分
data["A_total"] = A_items.mean(axis=1)
data["B_total"] = B_items.mean(axis=1)
data["C_total"] = C_items.mean(axis=1)

# 6. 相关性分析
corr_AB = data["A_total"].corr(data["B_total"])
corr_AC = data["A_total"].corr(data["C_total"])
corr_BC = data["B_total"].corr(data["C_total"])


# 7. EFA分析
def perform_efa(items, n_factors=1):
    scaler = StandardScaler()
    scaled_items = scaler.fit_transform(items)
    pca = PCA(n_components=n_factors)
    pca_scores = pca.fit_transform(scaled_items)
    combined = np.hstack((scaled_items, pca_scores))
    corr_matrix = np.corrcoef(combined, rowvar=False)
    loadings = corr_matrix[: scaled_items.shape[1], scaled_items.shape[1] :]
    return pca.explained_variance_ratio_[0], np.mean(np.abs(loadings))


efa_A_var, efa_A_loading = perform_efa(A_items)
efa_B_var, efa_B_loading = perform_efa(B_items)
efa_C_var, efa_C_loading = perform_efa(C_items)

# 8. 调节效应分析
model = smf.ols("B_total ~ A_total + C_total + A_total:C_total", data=data)
results = model.fit()
interaction_coef = results.params["A_total:C_total"]
interaction_pvalue = results.pvalues["A_total:C_total"]

# 9. 打印结果
print("===== 调节效应分析（核心验证） =====")
print(f"交互项 (A*C) 系数: {interaction_coef:.3f} (预期为负值)")
print(f"交互项显著性 p值: {interaction_pvalue:.5f}")
print()
print("===== 其他关键指标 =====")
print(f"A与B相关: {corr_AB:.3f} (预期负)")
print(f"B与C相关: {corr_BC:.3f} (预期负)")
print(f"A量表信度: {alpha_A:.3f}")
print(f"B量表信度: {alpha_B:.3f}")
print(f"C量表信度: {alpha_C:.3f}")


data.to_excel("501.xlsx", index=True)
