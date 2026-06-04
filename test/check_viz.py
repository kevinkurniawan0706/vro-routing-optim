import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 3.5, 2.5, 1]
y = [10, 15, 13, 18, 12, 13, 10]

# Create the plot
plt.scatter(x, y)
arrow_indices = [0, 1, 3, 0, 5, 2, 4, 0]

arrow_colors = [0, 0, 0, 1, 1, 1, 1]
color_mapping = {0: 'red', 1: 'green'}

# # Add arrows connecting points based on the arrow indices
# arrow_color = 'red'

# for i in range(len(arrow_indices) - 1):
#     start_index = arrow_indices[i]
#     end_index = arrow_indices[i + 1]

#     plt.annotate("",
#         xy=(x[end_index], y[end_index]),  # Endpoint of the arrow
#         xytext=(x[start_index], y[start_index]),  # Starting point of the arrow
#         arrowprops=dict(
#             arrowstyle="->",  # Arrow style (simple arrow)
#             color=arrow_color,  # Arrow color
#         )
#     )

# =====================================================================================

for i in range(len(arrow_indices) - 1):
    start_index = arrow_indices[i]
    end_index = arrow_indices[i + 1]
    arrow_color_numeric = arrow_colors[i]
    arrow_color = color_mapping.get(arrow_color_numeric, 'blue')  # Default to blue if not in mapping

    plt.annotate("",
        xy=(x[end_index], y[end_index]),  # Endpoint of the arrow
        xytext=(x[start_index], y[start_index]),  # Starting point of the arrow
        arrowprops=dict(
            arrowstyle="->",  # Arrow style (simple arrow)
            color=arrow_color,  # Arrow color
        )
    )


# Add labels and title (optional)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Simple Line Plot')

# Display the plot
plt.show()

