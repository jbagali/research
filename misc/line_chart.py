import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# Your data
x_values = [.1, .5, 1]
y_values = [.693, .823, .88]

y2_values = [.5567, .83, .88]
y3_values = [.0858, .3433, .47]


def percent_formatter(x, pos):
    """
    Formatter function to convert decimal to percentage on the y-axis.
    """
    return f'{x*100:.0f}%'


plt.figure(figsize=(4.5, 2.5))

# Plotting the points with visible dots
plt.scatter(x_values, y_values, color='blue', label='8-bit Adder', zorder=5)

# Plotting the line with dots
plt.plot(x_values, y_values, 'o--', color='blue')

plt.scatter(x_values, y2_values, color='red', label='8-bit Multiplier', zorder=5)
plt.plot(x_values, y2_values, 'o--', color='red')

plt.scatter(x_values, y3_values, color='green', label='8-bit MAC', zorder=5)
plt.plot(x_values, y3_values, 'o--', color='green')
# Adding labels and title
plt.xlabel('Baseline reward for functional Verilog codes')
plt.ylabel(r'% functional codes')
#plt.title('MCTS Baseline Reward v.s. Percent Functional Verilog Modules')

# Adding grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

plt.ylim(-0.1, 1)
plt.xticks(x_values)
# Adding legend
plt.legend(loc='lower right')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
plt.tight_layout()

plt.savefig("combined_reward_new.pdf", bbox_inches="tight")
# Show the plot