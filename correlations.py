import pandas as pd

# 读取 CSV 文件
file_path = './data/train.csv'
df = pd.read_csv(file_path)

# 计算皮尔逊相关系数矩阵
correlation_matrix = df.corr(method='pearson')
# 保存相关系数矩阵到 Excel 文件
output_file_path = 'correlation_matrix.xlsx'
correlation_matrix.to_excel(output_file_path, index=True)

print(f"相关系数矩阵已保存到 {output_file_path}")
