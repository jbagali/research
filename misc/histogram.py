import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# # Replace 'your_file.xlsx' with the actual path to your Excel file
# excel_file_path = 'Book1.xlsx'


# def count_values_in_window(array, window_size=50):
#     array_size = len(array)
#     num_windows = array_size // window_size
    
#     # Create an empty array to store the counts
#     counts_array = np.empty(num_windows, dtype=int)
    
#     for i in range(num_windows):
#         start_index = i * window_size
#         end_index = (i + 1) * window_size
#         window_values = array[start_index:end_index]
        
#         # Count non-NaN values in the window
#         counts_array[i] = np.sum(~np.isnan(window_values))
    
#     return counts_array


# # Read the Excel file into a pandas Series
# data_series = pd.read_excel(excel_file_path, header=None, squeeze=True)
# data_columns = data_series.iloc[:, :3]
# #data_sets = [column.tolist() for column in data_columns]
# numpy_array = data_columns.values
# array1 = numpy_array[:, 0]
# array2 = numpy_array[:, 1]
# array3 = numpy_array[:, 2]
# new_array1 = count_values_in_window(array1)
# new_array2 = count_values_in_window(array2)
# new_array3 = count_values_in_window(array3)
# print(new_array1)

# data = [new_array1, new_array2, new_array3]

# bin_labels = [f'{(i + 1) * 50}' for i in range(len(new_array1))]
# # Set up the figure and axis
# fig, ax = plt.subplots(figsize=(4.5,2.2))

# # Set the width of each bar
# bar_width = 0.25


# r1 = np.arange(len(new_array1))
# r2 = [x + bar_width for x in r1]
# r3 = [x + bar_width for x in r2]

# # Plot the histogram
# plt.bar(r1, data[0], color='#FFF2CC', width=bar_width, edgecolor='grey', label='4-bit MAC')
# plt.bar(r2, data[1], color='#DAE8FC', width=bar_width, edgecolor='grey', label='8-bit MAC')
# plt.bar(r3, data[2], color='#D5E8D4', width=bar_width, edgecolor='grey', label='16-bit MAC')

# # Add labels and title
# plt.xlabel('MCTS iteration number (50 iteration groups)')
# plt.ylabel('# of functional modules')
# #plt.title('Number of Functional Verilog Modules v.s. Number of MCTS Iterations')

# plt.xticks([r + bar_width for r in range(len(new_array1))], bin_labels)

# # Add legend
# plt.legend()
# plt.tight_layout()
# # Show the plot
# plt.savefig('mac_iterations_chart.pdf')








excel_file_path = 'combined_histogram_16bit.xlsx'


def count_values_in_window(array, window_size=50):
    array_size = len(array)
    num_windows = array_size // window_size
    
    # Create an empty array to store the counts
    counts_array = np.empty(num_windows, dtype=int)
    
    for i in range(num_windows):
        start_index = i * window_size
        end_index = (i + 1) * window_size
        window_values = array[start_index:end_index]
        
        # Count non-NaN values in the window
        counts_array[i] = np.sum(~np.isnan(window_values))
    
    return counts_array


# Read the Excel file into a pandas Series
data_series = pd.read_excel(excel_file_path, header=None, squeeze=True)
data_columns = data_series.iloc[:, :3]
#data_sets = [column.tolist() for column in data_columns]
numpy_array = data_columns.values
array1 = numpy_array[:, 0]
array2 = numpy_array[:, 1]
array3 = numpy_array[:, 2]
new_array1 = count_values_in_window(array1)
new_array2 = count_values_in_window(array2)
new_array3 = count_values_in_window(array3)
print(new_array1)

data = [new_array1, new_array2, new_array3]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4.5,2.2))

# Set the width of each bar
bar_width = 0.25


r1 = np.arange(len(new_array1))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot the histogram
plt.bar(r1, data[0], color='#FFF2CC', width=bar_width, edgecolor='grey', label='16-bit Adder')
plt.bar(r2, data[1], color='#DAE8FC', width=bar_width, edgecolor='grey', label='16-bit Multiplier')
plt.bar(r3, data[2], color='#D5E8D4', width=bar_width, edgecolor='grey', label='16-bit MAC')

# Add labels and title
plt.xlabel('MCTS iteration number')
plt.ylabel('# of functional codes')
#plt.title('Number of Functional Verilog Modules v.s. Number of MCTS Iterations')

bin_labels = [f'{(i + 1) * 50}' for i in range(len(new_array1))]
plt.xticks([r + bar_width for r in range(len(new_array1))], bin_labels)

# Add legend
plt.legend()
plt.tight_layout()
# Show the plot
plt.show()
plt.savefig('combined_histogram_result.pdf')