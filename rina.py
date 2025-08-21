import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, chi2
import statsmodels.api as sm

def calculate_bartlett_sphericity(data):
    n, p = data.shape
    R = np.corrcoef(data, rowvar=False)
    det_R = np.linalg.det(R)
    if det_R <= 0:
        det_R = 1e-10
    chi_square = - (n - 1 - (2*p + 5)/6) * np.log(det_R)
    df = p*(p-1)/2
    p_value = chi2.sf(chi_square, df)
    return chi_square, p_value

def calculate_kmo(data):
    R = np.corrcoef(data, rowvar=False)
    try:
        invR = np.linalg.inv(R + 1e-5 * np.eye(R.shape[0]))
    except np.linalg.LinAlgError:
        return None, None
    partial = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if i != j:
                partial[i,j] = -invR[i,j] / np.sqrt(invR[i,i] * invR[j,j])
    r2 = R**2
    p2 = partial**2
    np.fill_diagonal(r2, 0)
    np.fill_diagonal(p2, 0)
    sum_r2 = np.sum(r2)
    sum_p2 = np.sum(p2)
    kmo_model = sum_r2 / (sum_r2 + sum_p2)
    kmo_per = []
    for i in range(R.shape[0]):
        sum_r2_i = np.sum(r2[i,:])
        sum_p2_i = np.sum(p2[i,:])
        kmo_i = sum_r2_i / (sum_r2_i + sum_p2_i)
        kmo_per.append(kmo_i)
    return kmo_per, kmo_model

np.random.seed(42)
n = 500  # 样本量

# ========== 基础人口学题目 ==========
gender = np.random.choice([0, 1], size=n)  # 0=男 1=女
grade = np.random.choice([1, 2, 3], size=n)  # 大三/大四/大五

# ========== 潜变量生成 ==========
A1 = np.random.normal(3 + 0.3 * gender + 0.3 * (grade - 2), 0.6, n)
A2 = np.random.normal(3 - 0.2 * gender + 0.2 * (grade - 2), 0.6, n)
D1 = -0.5 * A1 + 0.4 * A2 + np.random.normal(0, 0.4, n)
D2 = -0.5 * A1 + 0.4 * A2 + np.random.normal(0, 0.4, n)
D3 = -0.5 * A1 + 0.4 * A2 + np.random.normal(0, 0.4, n)
B1 = -0.6 * A1 + 0.5 * A2 + 0.5 * D1 + np.random.normal(0, 0.4, n)
B2 = -0.6 * A1 + 0.5 * A2 + 0.5 * D2 + np.random.normal(0, 0.4, n)
B3 = -0.6 * A1 + 0.5 * A2 + 0.5 * D3 + np.random.normal(0, 0.4, n)
B4 = -0.6 * A1 + 0.5 * A2 + 0.5 * D1 + np.random.normal(0, 0.4, n)
B5 = -0.6 * A1 + 0.5 * A2 + 0.5 * D2 + np.random.normal(0, 0.4, n)
C1 = -0.8 * A1 - 0.6 * A2 - 0.4 * D1 - 0.5 * B1 + 0.15 * gender + np.random.normal(0, 0.3, n)
C2 = -0.8 * A1 - 0.6 * A2 - 0.4 * D2 - 0.5 * B2 + 0.15 * gender + np.random.normal(0, 0.3, n)

# ========== 缩放到 1-5 Likert ==========
def to_likert(x):
    return np.clip(np.round((x - np.min(x)) / (np.max(x) - np.min(x)) * 4 + 1), 1, 5)

# 生成各小题
A1_items = np.array([to_likert(A1 + np.random.normal(0, 0.3, n)) for _ in range(5)]).T  # q3-q7
A2_items = np.array([to_likert(A2 + np.random.normal(0, 0.3, n)) for _ in range(5)]).T  # q8-q12
B1_items = np.array([to_likert(B1 + np.random.normal(0, 0.3, n)) for _ in range(5)]).T  # q13-q17
B2_items = np.array([to_likert(B2 + np.random.normal(0, 0.3, n)) for _ in range(6)]).T  # q18-q23
B3_items = np.array([to_likert(B3 + np.random.normal(0, 0.3, n)) for _ in range(4)]).T  # q24-q27
B4_items = np.array([to_likert(B4 + np.random.normal(0, 0.3, n)) for _ in range(6)]).T  # q28-q33
B5_items = np.array([to_likert(B5 + np.random.normal(0, 0.3, n)) for _ in range(6)]).T  # q34-q39
C1_items = np.array([to_likert(C1 + np.random.normal(0, 0.3, n)) for _ in range(7)]).T  # q40-q46
C2_items = np.array([to_likert(C2 + np.random.normal(0, 0.3, n)) for _ in range(6)]).T  # q47-q52
D1_items = np.array([to_likert(D1 + np.random.normal(0, 0.3, n)) for _ in range(3)]).T  # q53-q55
D2_items = np.array([to_likert(D2 + np.random.normal(0, 0.3, n)) for _ in range(4)]).T  # q56-q59
D3_items = np.array([to_likert(D3 + np.random.normal(0, 0.3, n)) for _ in range(5)]).T  # q60-q64

# ========== 组合成 DataFrame ==========
columns = ['gender', 'grade'] + [f'q{i}' for i in range(3, 65)]  # 修正：q3-q64，共64列
data = np.hstack([
    gender.reshape(-1, 1),
    grade.reshape(-1, 1),
    A1_items, A2_items,
    B1_items, B2_items, B3_items, B4_items, B5_items,
    C1_items, C2_items,
    D1_items, D2_items, D3_items
])
df = pd.DataFrame(data, columns=columns)
df = df.astype(int)  # 确保所有数据为整数

# 保存到CSV
# df.to_csv("data.csv", index=False)

# ========== 计算子维度均值 ==========
df['A1'] = df[[f'q{i}' for i in range(3, 8)]].mean(axis=1)
df['A2'] = df[[f'q{i}' for i in range(8, 13)]].mean(axis=1)
df['B1'] = df[[f'q{i}' for i in range(13, 18)]].mean(axis=1)
df['B2'] = df[[f'q{i}' for i in range(18, 24)]].mean(axis=1)
df['B3'] = df[[f'q{i}' for i in range(24, 28)]].mean(axis=1)
df['B4'] = df[[f'q{i}' for i in range(28, 34)]].mean(axis=1)
df['B5'] = df[[f'q{i}' for i in range(34, 40)]].mean(axis=1)
df['C1'] = df[[f'q{i}' for i in range(40, 47)]].mean(axis=1)
df['C2'] = df[[f'q{i}' for i in range(47, 53)]].mean(axis=1)
df['D1'] = df[[f'q{i}' for i in range(53, 56)]].mean(axis=1)
df['D2'] = df[[f'q{i}' for i in range(56, 60)]].mean(axis=1)
df['D3'] = df[[f'q{i}' for i in range(60, 65)]].mean(axis=1)
df['B'] = df[[f'q{i}' for i in range(13, 40)]].mean(axis=1)
df['C'] = df[[f'q{i}' for i in range(40, 53)]].mean(axis=1)
df['D'] = df[[f'q{i}' for i in range(53, 65)]].mean(axis=1)

# ========== 信度分析（Cronbach's α） ==========
def cronbach_alpha(items):
    items = np.array(items)
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    n_items = items.shape[1]
    return n_items / (n_items - 1) * (1 - item_vars.sum() / total_var)

print("Cronbach's α:")
for dim in ['A1', 'A2', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'D1', 'D2', 'D3', 'B', 'C', 'D']:
    items = df[[f'q{i}' for i in range(*{'A1': (3,8), 'A2': (8,13), 'B1': (13,18), 'B2': (18,24), 'B3': (24,28), 
                                         'B4': (28,34), 'B5': (34,40), 'C1': (40,47), 'C2': (47,53), 
                                         'D1': (53,56), 'D2': (56,60), 'D3': (60,65), 
                                         'B': (13,40), 'C': (40,53), 'D': (53,65)}[dim])]]
    print(f"{dim}: {cronbach_alpha(items):.3f}")

# ========== KMO & Bartlett ==========
all_items = df[[f'q{i}' for i in range(3, 65)]].values
kmo_all, kmo_model = calculate_kmo(all_items)
chi_square_value, p_value = calculate_bartlett_sphericity(all_items)
print("\nKMO:", kmo_model)
print("Bartlett p-value:", p_value)

# ========== 假设检验（相关性） ==========
pairs = [("A1", "C"), ("A2", "C"), ("A1", "D"), ("A2", "D"),
         ("A1", "B"), ("A2", "B"), ("D", "C"), ("D", "B"), ("B", "C")]

print("\n相关性检验：")
for x, y in pairs:
    r, p = pearsonr(df[x], df[y])
    print(f"{x} vs {y}: r={r:.3f}, p={p:.3e}")

# ========== T检验（性别） ==========
print("\n独立样本 T 检验（性别）：")
for var in ["A1", "A2", "B", "C", "D"]:
    t, p = ttest_ind(df[var][df['gender']==0], df[var][df['gender']==1])
    print(f"{var} by gender: t={t:.3f}, p={p:.3e}")

# ========== 年级 ANOVA（OLS方式） ==========
print("\n年级对 A1 影响的 ANOVA：")
X = pd.get_dummies(df['grade'], drop_first=True).astype(float)
X = sm.add_constant(X)
y = df['A1'].astype(float)
model = sm.OLS(y, X).fit()
print(model.summary())

# ========== 回归（A 对 C） ==========
print("\n多元回归：C ~ A1 + A2")
X = df[["A1", "A2"]]
X = sm.add_constant(X)
y = df["C"]
model = sm.OLS(y, X).fit()
print(model.summary())

# ========== 中介作用检验 ==========
print("\n中介作用检验：")
# D as mediator
print("D ~ A1 + A2")
X_d = df[["A1", "A2"]]
X_d = sm.add_constant(X_d)
y_d = df["D"]
model_d = sm.OLS(y_d, X_d).fit()
print(model_d.summary())

print("C ~ A1 + A2 + D")
X_with_d = df[["A1", "A2", "D"]]
X_with_d = sm.add_constant(X_with_d)
model_with_d = sm.OLS(y, X_with_d).fit()
print(model_with_d.summary())

# B as mediator
print("B ~ A1 + A2")
X_b = df[["A1", "A2"]]
X_b = sm.add_constant(X_b)
y_b = df["B"]
model_b = sm.OLS(y_b, X_b).fit()
print(model_b.summary())

print("C ~ A1 + A2 + B")
X_with_b = df[["A1", "A2", "B"]]
X_with_b = sm.add_constant(X_with_b)
model_with_b = sm.OLS(y, X_with_b).fit()
print(model_with_b.summary())

# Chain: D and B
print("C ~ A1 + A2 + D + B")
X_with_db = df[["A1", "A2", "D", "B"]]
X_with_db = sm.add_constant(X_with_db)
model_with_db = sm.OLS(y, X_with_db).fit()
print(model_with_db.summary())
