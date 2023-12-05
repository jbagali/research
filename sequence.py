import pandas as pd

# Step 1: Read Excel file into a pandas DataFrame
excel_file_path = 'sequence.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(excel_file_path, header=None, names=['data_column'])

# Step 2: Find the longest sequence of any single numerical value excluding -0.1 and -1
max_sequence_length = 0
current_sequence_length = 1
current_value = df['data_column'][0]
excluded_values = [-0.1, -.75, -1]

for i in range(1, len(df)):
    if df['data_column'][i] == current_value and df['data_column'][i]:
        current_sequence_length += 1
    else:
        if current_sequence_length > max_sequence_length:
            max_sequence_length = current_sequence_length
        current_sequence_length = 1
        current_value = df['data_column'][i]

# Check the last sequence
if current_sequence_length > max_sequence_length:
    max_sequence_length = current_sequence_length

# Step 3: Print or use the result as needed
print(f"The longest sequence of any single numerical value (excluding -0.1 and -1) is: {max_sequence_length}")