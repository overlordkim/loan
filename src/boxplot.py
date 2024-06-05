import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column = 'f4'

# 读取CSV文件
csv_file_path = '../assets/train_public.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 过滤掉 NaN 值
non_nan = df[column].dropna()

# 设置图的大小
plt.figure(figsize=(12, 8))

# 绘制箱线图，并设置宽度
sns.boxplot(x=non_nan, color='skyblue', width=0.5)

# 设置标题和标签
plt.title('Box Plot of ' + column, fontsize=16)
plt.xlabel(column, fontsize=14)

# 显示网格（可选）
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 保存图像
output_path = '../graphs/' + column + '_boxplot.png'
plt.tight_layout()
plt.savefig(output_path, format='png')
plt.close()  # 关闭图像显示

# 计算四分位数和四分位距
Q1 = non_nan.quantile(0.25)
Q3 = non_nan.quantile(0.75)
IQR = Q3 - Q1

# 计算上限
upper_limit = Q3 + 1.5 * IQR

# 找出大于上限的值
outliers = non_nan[non_nan > upper_limit]

# 打印统计信息
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Upper limit for outliers: {upper_limit}")
print(f"Number of outliers: {len(outliers)}")
print(f"Outliers: {outliers.values}")

# 画出概率密度和直方图（上面的代码保持不变）
plt.figure(figsize=(12, 8))
sns.histplot(non_nan, kde=True, color='skyblue', bins=30)

plt.title('Probability Density and Histogram of ' + column, fontsize=16)
plt.xlabel(column, fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_path = '../graphs/' + column + '_density_histogram.png'
plt.tight_layout()
plt.savefig(output_path, format='png')
plt.close()
print(f"Probability density and histogram saved to {output_path}")