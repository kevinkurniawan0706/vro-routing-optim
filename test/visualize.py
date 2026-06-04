import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt

import os
import pickle

# Menyusun path absolut berdasarkan lokasi file visualize.py ini berada
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'data', 'hcvrp', 'custom_data_tw.pkl')
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)
# Data pada index [0] karena kita melihat hasil evaluasi baris pertama (tour[0])
print("Depot Coords: ", data[0][0])
print("Customer Coords: ", data[0][1])
print("Demands Coords: ", data[0][2])
print("Kapasitas Tipe Mobil Coords: ", data[0][3])
# print("Time Windows Coords: ", data[0][4])

# Coordinate of depot and customers
# depot_coords = [0.686477633387903, 0.14976072122002826]
# customer_coords = [[0.795757990675748, 0.5893125789478779], [0.2532809231139508, 0.7479129904242673], [0.8642762990940475, 0.1316337043138731], [0.5714687407684512, 0.8373812335316075], [0.3884959466940112, 0.3184581315185151], [0.18419639770208862, 0.1553452926029033], [0.411230882622858, 0.196152162818097], [0.6011834427418628, 0.8706626130536566], [0.06960650008954472, 0.47712141397827357], [0.03260846045045618, 0.5447878685201997], [0.3763150870988762, 0.12102769789714918], [0.39646244096227945, 0.6032844405417088], [0.2060989043753021, 0.6676390476348506], [0.38460152976412, 0.6357962114143128], [0.22593840447495905, 0.7580562404221568], [0.4767740889830774, 0.4204432964005559], [0.8554383062061905, 0.29812094777448417], [0.725027483971654, 0.26995987667144705], [0.3895860619020679, 0.0754289167555503], [0.9466116840511963, 0.35844843919662805], [0.9628522783257579, 0.29462108191397374], [0.8751448195075507, 0.601489776756959], [0.1710110942377222, 0.5168428086283081], [0.7631135704493942, 0.6777854570103262], [0.17667730289174954, 0.7348620409438535], [0.3535472313915995, 0.9130310932186962], [0.4805619948724553, 0.8575939281865703], [0.8830865864427092, 0.8071443950710563], [0.6361892775621253, 0.6494937756660057], [0.3289603296009491, 0.8379477289428744], [0.5414605796172374, 0.5965653452245552], [0.06997134440540764, 0.535009126362742], [0.2817893242643601, 0.5456070139710086], [0.6669090217065761, 0.0849233299334029], [0.9218200609095217, 0.000129565814382393], [0.22984526321748955, 0.6243008775040917], [0.5983078653682742, 0.47628154634670417], [0.4084596343859984, 0.9889039831000234], [0.5434976057172338, 0.7726847452854668], [0.7736040949252803, 0.5838220616706873]]

depot_coords = data[0][0]
customer_coords = data[0][1]

# tour = [6, 13, 11, 15, 40, 7, 27, 28, 0, 38, 25, 22, 5, 24, 23, 0, 19, 35, 39, 36, 2, 0, 21, 26, 0, 1, 14, 4, 33, 12, 0, 0, 34, 18, 0, 37, 0, 16, 0, 8, 0, 0, 17, 0, 10, 0, 29, 0, 30, 0, 9, 20, 0, 0, 31, 32, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# tour = [29, 0, 11, 23, 9, 6, 37, 0, 7, 0, 19, 34, 18, 0, 3, 35, 17, 0, 21, 20, 1, 0, 40, 22, 28, 24, 8, 4, 5, 0, 39, 27, 38, 26, 30, 31, 0, 2, 25, 15, 13, 16, 0, 14, 36, 32, 10, 12, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# tour = [29, 0, 11, 23, 9, 6, 37, 0, 7, 0, 19, 34, 18, 0, 3, 35, 17, 0, 21, 20, 1, 0, 40, 22, 28, 24, 8, 4, 5, 0, 39, 27, 38, 26, 30, 31, 0, 2, 25, 15, 13, 16, 0, 14, 36, 32, 10, 12, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tour = [33, 0, 12, 0, 23, 26, 11, 15, 13, 18, 17, 24, 21, 16, 28, 29, 22, 14, 19, 25, 20, 27, 10, 9, 35, 32, 37, 34, 31, 36, 2, 4, 1, 3, 30, 7, 0, 5, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# vehicle_tours = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# vehicle_tours=[0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
vehicle_tours = [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

color_mapping = {0: 'red', 1: 'green', 2: 'blue'}

x = [i[0] for i in customer_coords]
y = [i[1] for i in customer_coords]

# Create the plot
x.insert(0, depot_coords[0])
y.insert(0, depot_coords[1])

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Create a figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Setup subplots
ax_main = fig.add_subplot(gs[0, :])
ax_v = [fig.add_subplot(gs[1, i]) for i in range(3)]
axes = [ax_main] + ax_v

# Helper function untuk menggambar titik depot dan customer di suatu axes (grafik)
def plot_base_nodes(ax):
    # Plot depot
    ax.scatter(depot_coords[0], depot_coords[1], c='red', marker='s', s=80, label='Depot (0)', zorder=5)
    
    # Plot customers
    customer_nums = range(1, len(customer_coords) + 1)
    for num, (cx, cy) in zip(customer_nums, customer_coords):
        ax.scatter(cx, cy, c='yellow', edgecolors='gray', marker='o', zorder=4)
        ax.text(cx, cy, str(num), fontsize=8, ha='center', va='center', zorder=6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Helper function untuk menggambar rute suatu tipe mobil di axes tertentu
def draw_route(ax, v_type_color, v_tour):
    for i in range(len(v_tour) - 1):
        start_index = v_tour[i]
        end_index = v_tour[i + 1]
        
        if start_index == 0 and end_index == 0:
            continue
            
        is_return = (end_index == 0)
        alpha_val = 0.3 if is_return else 0.8
        ls_val = "dashed" if is_return else "solid"

        ax.annotate("",
            xy=(x[end_index], y[end_index]),  
            xytext=(x[start_index], y[start_index]),  
            arrowprops=dict(
                arrowstyle="->",
                color=v_type_color,
                alpha=alpha_val,
                ls=ls_val,
                connectionstyle="arc3,rad=0.1" 
            ),
            zorder=3
        )

# Gambar base nodes pada semua grafik
for ax in axes:
    plot_base_nodes(ax)

ax_main.set_title('Grafik Overall (Semua Tipe Mobil)', fontweight='bold')

# Gambar garis rute
for v_type in sorted(color_mapping.keys()):
    v_tour = [0] + [tour[i] for i in range(len(tour)) if i < len(vehicle_tours) and vehicle_tours[i] == v_type]
    
    # RL environment secara implisit menyuruh mobil pulang ke markas di akhir proses
    if len(v_tour) > 0 and v_tour[-1] != 0:
        v_tour.append(0)
        
    arrow_color = color_mapping[v_type]
    
    # Gambar di grafik utama
    draw_route(ax_main, arrow_color, v_tour)
    
    # Gambar di grafik spesifik tipe mobil tersebut
    ax_v[v_type].set_title(f'Tipe Mobil {v_type + 1} ({arrow_color.upper()})')
    draw_route(ax_v[v_type], arrow_color, v_tour)

# Legend untuk grafik utama
custom_lines = [Line2D([0], [0], color=color_mapping[k], lw=2) for k in sorted(color_mapping.keys())]
ax_main.legend(custom_lines, [f'Tipe Mobil {k+1}' for k in sorted(color_mapping.keys())])

# Cetak Breakdown Rute ke Terminal
print("\n=== BREAKDOWN RUTE PER TIPE MOBIL ===")
for v_type in sorted(color_mapping.keys()):
    v_tour = [0] + [tour[i] for i in range(len(tour)) if i < len(vehicle_tours) and vehicle_tours[i] == v_type]
    
    if len(v_tour) > 0 and v_tour[-1] != 0:
        v_tour.append(0)
        
    cleaned_tour = []
    for i in range(len(v_tour)):
        if i > 0 and v_tour[i] == 0 and v_tour[i-1] == 0:
            continue
        cleaned_tour.append(v_tour[i])
        
    route_str = " -> ".join(["Depot" if n == 0 else str(n) for n in cleaned_tour])
    print(f"Tipe Mobil {v_type + 1} ({color_mapping[v_type].upper()}):")
    print(route_str)
    print("-" * 50)

plt.tight_layout()
plt.show()

