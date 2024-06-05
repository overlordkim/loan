import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column = 'f2'

# 读取CSV文件
csv_file_path = '../assets/train_public.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file_path)

# 过滤掉 NaN 值
non_nan = df[column].dropna()

# 绘制概率密度图（KDE）和直方图
plt.figure(figsize=(12, 8))
sns.histplot(non_nan, kde=True, color='skyblue', bins=30)

# 设置标题和标签
plt.title('Probability Density and Histogram of ' + column, fontsize=16)
plt.xlabel(column, fontsize=14)
plt.ylabel('Density', fontsize=14)

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图像
output_path = '../graphs/' + column + '_density_histogram.png'
plt.tight_layout()
plt.savefig(output_path, format='png')
plt.close()  # 关闭图像显示

# 打印提示信息
print(f"Probability density and histogram saved to {output_path}")