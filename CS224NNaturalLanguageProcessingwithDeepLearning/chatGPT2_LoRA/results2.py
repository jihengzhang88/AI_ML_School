import matplotlib.pyplot as plt

# Define the given points
points = [
    (0, 0.632), (0.1, 0.632), (0.2, 0.632), (0.3, 0.633), 
    (0.4, 0.636), (0.5, 0.641), (0.6, 0.651), (0.7, 0.682), 
    (0.8, 0.750), (0.9, 0.813), (1.0, 0.829)
]

# Extract x and y coordinates
x_values, y_values = zip(*points)

# Baseline accuracy point
baseline_x = 0.5
baseline_y = 0.829

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5, label="LoRA Weight merging")

# Add baseline accuracy as a red dot
plt.scatter(baseline_x, baseline_y, color='r', s=50, label="Full model tuning Baseline")
plt.text(baseline_x, baseline_y + 0.005, "Baseline Accuracy", color='r', fontsize=10, ha='right')

# Customize the plot
plt.xlabel("Paraphrase Weight")
plt.ylabel("Paraphrase Accuracy")
plt.title("Weight merging Paraphrase and CIFAR test on Paraphrase")
plt.grid(True)
plt.legend()

# Set x-axis tick marks at every 0.1
plt.xticks([i / 10 for i in range(11)])

# Show the plot
plt.show()