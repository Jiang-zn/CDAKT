# 定义要分割的文件名和要生成的文件数量
input_file = 'assist2015_valid21.csv'
output_files = ['assist2015_valid1.csv', 'assist2015_valid2.csv', 'assist2015_valid3.csv', 'assist2015_valid4.csv']
num_output_files = len(output_files)

# 打开源文件以读取数据
with open(input_file, 'r') as source_file:
    # 读取源文件的所有行
    lines = source_file.readlines()

# 计算每个输出文件应包含的行数（近似等份，且可以被3整除）
total_lines = len(lines)
lines_per_output_file = total_lines // num_output_files
lines_per_output_file -= lines_per_output_file % 3  # 确保每份可以被3整除
# 将数据分割并写入不同的输出文件
for i, output_file in enumerate(output_files):
    # 计算要写入的行范围
    start_line = i * lines_per_output_file
    end_line = start_line + lines_per_output_file

    # 打开输出文件以写入数据
    with open(output_file, 'w') as target_file:
        # 使用切片操作将数据写入输出文件
        target_file.writelines(lines[start_line:end_line])

print(f'分割完成，生成了 {num_output_files} 个文件。')


# import csv
#
# # 指定文件数量
# file_count = 20
# tes='assist2015_valid'
# # 创建CSV文件
# for i in range(1, file_count + 1):
#     # 生成文件名，例如：1.csv, 2.csv, ..., 20.csv
#     file_name = f'{tes}{i}.csv'
#
#     # 使用空白数据创建CSV文件
#     with open(file_name, 'w', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         # 可以选择写入文件头部（列名）或者留空
#         # csvwriter.writerow(['Column1', 'Column2', 'Column3'])  # 如果需要写入列名，请取消注释这一行
#
# # 提示：上面的代码会在当前工作目录创建20个空的CSV文件，如果需要在文件中写入特定的列名，请取消注释相应行，并修改列名为您需要的内容。
# print(f'成功创建 {file_count} 个CSV文件。')