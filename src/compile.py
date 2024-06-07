import pandas as pd

def copy_loan_id_to_new_csv(first_csv, second_csv, output_csv):
    # 读取第一个CSV文件
    first_df = pd.read_csv(first_csv)
    
    # 检查是否包含loan_id列
    if 'loan_id' not in first_df.columns:
        raise ValueError("The first CSV file does not contain a 'loan_id' column.")
    
    # 读取第二个CSV文件
    second_df = pd.read_csv(second_csv)
    
    # 获取第一个CSV文件中的loan_id列
    loan_id_column = first_df['loan_id']
    
    # 将loan_id列添加到第二个CSV文件的第一列，并改名为id
    second_df.insert(0, 'id', loan_id_column)
    
    # 保存新的CSV文件
    second_df.to_csv(output_csv, index=False)
    print(f'New CSV file saved to {output_csv}')

# 使用示例
first_csv = '../assets/test_public.csv'
second_csv = '../assets/submission.csv'
output_csv = '../assets/submission.csv'
copy_loan_id_to_new_csv(first_csv, second_csv, output_csv)
