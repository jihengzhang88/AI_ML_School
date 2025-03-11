import matplotlib.pyplot as plt

# Define the given points
points = [
    (0, 0.457), (0.1, 0.451), (0.2, 0.441), (0.3, 0.451), 
    (0.4, 0.462), (0.5, 0.468), (0.6, 0.475), (0.7, 0.490), 
    (0.8, 0.505), (0.9, 0.512), (1.0, 0.520)
]

# Extract x and y coordinates
x_values, y_values = zip(*points)

# Baseline accuracy point
baseline_x = 0.5
baseline_y = 0.490

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5, label="LoRA Weight merging")

# Add baseline accuracy as a red dot
plt.scatter(baseline_x, baseline_y, color='r', s=50, label="Full model tuning Baseline")
plt.text(baseline_x, baseline_y + 0.005, "Baseline Accuracy", color='r', fontsize=10, ha='right')

# Customize the plot
plt.xlabel("SST Weight")
plt.ylabel("SST Accuracy")
plt.title("Weight merging SST and CIFAR test on SST")
plt.grid(True)
plt.legend()

# Set x-axis tick marks at every 0.1
plt.xticks([i / 10 for i in range(11)])

# Show the plot
plt.show()