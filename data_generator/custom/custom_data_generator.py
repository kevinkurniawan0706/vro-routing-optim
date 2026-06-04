import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import numpy as np
import pickle
from utils.data_utils import save_dataset

def create_custom_dataset(save_path, json_path, veh_num=3):
    """
    Fungsi ini membaca data asli (depot, kordinat customer, demand) dari file JSON,
    lalu memformatnya menjadi dataset .pkl yang dapat dievaluasi oleh model HCVRPTW.
    """
    
    # --- BACA DATA JSON ---
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File konfigurasi JSON tidak ditemukan di: {json_path}")
        
    with open(json_path, 'r') as f:
        payload = json.load(f)
        
    raw_depot_coords = payload['raw_depot_coords']
    raw_customer_coords = payload['raw_customer_coords']
    customer_demands = payload['customer_demands']
    tw_start = payload['time_window_default']['start_hour']
    tw_end = payload['time_window_default']['end_hour']
    
    num_customers = len(raw_customer_coords)
    
    # Pastikan jumlah customer sama dengan jumlah demand
    if len(customer_demands) != num_customers:
        raise ValueError(f"Jumlah demand ({len(customer_demands)}) tidak sama dengan jumlah customer ({num_customers})!")
    
    # === NORMALISASI KOORDINAT [0, 1] ===
    # Model HCVRP terbiasa dengan koordinat di rentang 0 hingga 1.
    all_coords = np.array([raw_depot_coords] + raw_customer_coords)
    min_val = np.min(all_coords, axis=0)
    max_val = np.max(all_coords, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-5 # Menghindari pembagian dengan 0
    
    norm_coords = (all_coords - min_val) / range_val
    depot_coords = norm_coords[0].tolist()
    customer_coords = norm_coords[1:].tolist()
    
    # --- KAPASITAS KENDARAAN ---
    if veh_num == 3:
        vehicle_capacities = [6.2, 3.8, 2.2]
    elif veh_num == 5:
        vehicle_capacities = [20.0, 25.0, 30.0, 35.0, 40.0]
    else:
        raise ValueError("Script ini diatur untuk 3 atau 5 kendaraan.")

    # --- TIME WINDOWS (Jam Operasional) ---
    # Matrix 2D berukuran (num_customers x 24). 
    time_windows = np.zeros((num_customers, 24))
    
    for i in range(num_customers):
        time_windows[i, tw_start:tw_end] = 1
        
    # --- PROSES PACKING DATA ---
    dataset = []
    
    instance = (
        depot_coords,
        customer_coords,
        customer_demands,
        vehicle_capacities,
        time_windows.tolist()
    )
    
    dataset.append(instance)
    
    # --- PENYIMPANAN KE .PKL ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dataset(dataset, save_path)
    print(f"Custom dataset berhasil disimpan di: {save_path}")
    print(f"Jumlah instance: {len(dataset)}")
    print(f"Jumlah customer per instance: {num_customers}")


if __name__ == "__main__":
    # Path ke file pkl output
    output_filename = "data/hcvrp/custom_data_tw.pkl"
    
    # Path ke file json yang se-folder dengan script ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_payload_path = os.path.join(current_dir, "custom_data_payload.json")
    
    # Panggil fungsi
    create_custom_dataset(output_filename, json_payload_path, veh_num=3)
