import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt

# Coordinate of depot and customers
depot_coords = [0.686477633387903, 0.14976072122002826]
customer_coords = [
    [0.795757990675748, 0.5893125789478779], [0.2532809231139508, 0.7479129904242673], 
    [0.8642762990940475, 0.1316337043138731], [0.5714687407684512, 0.8373812335316075], 
    [0.3884959466940112, 0.3184581315185151], [0.18419639770208862, 0.1553452926029033], 
    [0.411230882622858, 0.196152162818097], [0.6011834427418628, 0.8706626130536566], 
    [0.06960650008954472, 0.47712141397827357], [0.03260846045045618, 0.5447878685201997], 
    [0.3763150870988762, 0.12102769789714918], [0.39646244096227945, 0.6032844405417088], 
    [0.2060989043753021, 0.6676390476348506], [0.38460152976412, 0.6357962114143128], 
    [0.22593840447495905, 0.7580562404221568], [0.4767740889830774, 0.4204432964005559], 
    [0.8554383062061905, 0.29812094777448417], [0.725027483971654, 0.26995987667144705], 
    [0.3895860619020679, 0.0754289167555503], [0.9466116840511963, 0.35844843919662805], 
    [0.9628522783257579, 0.29462108191397374], [0.8751448195075507, 0.601489776756959], 
    [0.1710110942377222, 0.5168428086283081], [0.7631135704493942, 0.6777854570103262], 
    [0.17667730289174954, 0.7348620409438535], [0.3535472313915995, 0.9130310932186962], 
    [0.4805619948724553, 0.8575939281865703], [0.8830865864427092, 0.8071443950710563], 
    [0.6361892775621253, 0.6494937756660057], [0.3289603296009491, 0.8379477289428744], 
    [0.5414605796172374, 0.5965653452245552], [0.06997134440540764, 0.535009126362742], 
    [0.2817893242643601, 0.5456070139710086], [0.6669090217065761, 0.0849233299334029], 
    [0.9218200609095217, 0.000129565814382393], [0.22984526321748955, 0.6243008775040917], 
    [0.5983078653682742, 0.47628154634670417], [0.4084596343859984, 0.9889039831000234], 
    [0.5434976057172338, 0.7726847452854668], [0.7736040949252803, 0.5838220616706873]
]
for i in customer_coords:
    print(i)
    print(f"x: {i[0]}, y: {i[1]}")
x = [i[0] for i in customer_coords]
print(x)
y = [i[1] for i in customer_coords]
print(y)
# tour = [11, 6, 19, 7, 0, 35, 3, 0, 21, 20, 22, 1, 0, 9, 10, 32, 16, 23, 13, 25, 36, 0, 2, 15, 30, 26, 27, 39, 0, 28, 24, 8, 38, 12, 0, 0, 40, 4, 29, 5, 33, 14, 31, 37, 18, 0, 17, 0, 0, 34, 0, 0]
# vehicle_tours = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 1, 1, 2, 1, 0, 2, 1, 2, 1, 1, 0]

tour = [11, 6, 19, 7, 0]
vehicle_tours = [1, 1, 1, 1, 1]

color_mapping = {0: 'red', 1: 'green', 2: 'brown'}
# Create the plot
x.insert(0, depot_coords[0])
y.insert(0, depot_coords[1])

plt.scatter(x, y)
plt.scatter(depot_coords[0], depot_coords[1], c='red', marker='s', label='Depot (0)')

# Add an arrow annotation
# label = 3  # Index of the point where you want to add the arrow
# prev_label = 2  # Index of the previous point
# arrow_color = 'red'


for i in range(len(tour) - 1):
    start_index = tour[i]
    end_index = tour[i + 1]
    arrow_color_numeric = vehicle_tours[i]
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


