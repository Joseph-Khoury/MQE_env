# Mach-Zehnder Interferometer Interactive Simulation Assignment
# For: Quantum Engineering MSc, Semester 2
# Task: Rebuild this interactive simulation step-by-step with guidance

# --- Imports ---
# TODO: Import numpy as np (used for array calculations)
# TODO: Import matplotlib.pyplot as plt (used for plotting)
# TODO: Import Slider from matplotlib.widgets (used for interactivity)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c, epsilon_0

# --- Constants ---
# Define light wavelength (e.g., 500 nm)
wavelength = 500 * 1e-9
# Compute wave number k = 2*pi / wavelength
k = 2*np.pi / wavelength

# Fixed source position
z1 = 2 #mm # TODO 
# Simulation grid size (e.g., 500 x 500)
grid_size = 500
# Size of the observation window (e.g., 5 mm)
L = 5
# TODO: Create x and y coordinate arrays using np.linspace
x = np.linspace(-L/2,L/2, grid_size)
y = np.linspace(-L/2,L/2, grid_size)
# TODO: Create 2D grid using np.meshgrid
X, Y = np.meshgrid(x,y)

# --- Function: Compute Classical Interference ---
def compute_classical_intensity(z2):
    # Compute optical path lengths
    R1 = np.sqrt(X**2 + Y**2 + z1**2) # TODO 
    R2 = np.sqrt(X**2 + Y**2 + z2**2)# TODO

    # Compute fields from both paths
    E1 = (1/R1) * np.exp(1j * k * R1) # TODO
    E2 = (1/R2) * np.exp(1j * k * R2) # TODO

    # Add electric fields
    E_total = E1 + E2 # TODO

    # Compute intensity and normalize
    I = c*epsilon_0/2 * np.pow(abs(E_total),2) # TODO
    return I # TODO

# --- Function: Simulate Photon Detection ---
# def simulate_quantum_hits(z2, num_photons):
#     num_photons = int(num_photons)
#     # Get classical intensity
#     I_classical = compute_classical_intensity(z2)  # Use your classical intensity function to get 2D intensity map
#     prob_dist = I_classical / I_classical.sum()  # Normalize to create probability distribution
#     I_flat = prob_dist.ravel()  # Flatten for sampling
#     shape = prob_dist.shape
    
#     # TODO: Sample photon_indices from the intensity-based probability distribution
#     photon_indices = np.random.choice(a=np.arange(I_flat.size), size=num_photons, p=I_flat) # TODO: Use np.random.choice with appropriate arguments
    
#     hit_counts = np.bincount(photon_indices, minlength=I_flat.size)
    
#     hit_map = hit_counts.reshape(shape)
    
#     # Normalize for display
#     if hit_map.max() > 0:
#         hit_map = hit_map / hit_map.max()
#     # coords = np.array(np.unravel_index(np.arange(I_flat.size), prob_dist.shape)).T  # Map indices to 2D coords

#     # print(len(photon_indices))
#     # photon_coords = coords[photon_indices]  # Get 2D coordinates for hits
#     # hit_map = np.zeros_like(prob_dist)  # Initialize hit map

#     # # TODO: Increment hit map based on sampled photon coordinates
#     # for idx in photon_coords:
#     #     hit_map[tuple(idx)] += 1

#     return hit_map

import psutil
import time
from joblib import Parallel, delayed, cpu_count

def _sample_photons(I_flat, num_photons):
    indices = np.random.choice(I_flat.size, size=num_photons, p=I_flat)
    return np.bincount(indices, minlength=I_flat.size)

def estimate_safe_batch_size(target_memory_gb=2.0, photons_per_mb=1_000_000):
    """Estimate a safe batch size based on available memory (default: use 2GB)."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    safe_gb = min(target_memory_gb, available_gb)  # Be conservative
    batch_size = int(safe_gb * 1024 * 1024 / (4))  # Estimate: 4 bytes per photon (float32/uint32)
    return max(batch_size, photons_per_mb * 10)  # Ensure at least 2.5M photons

def simulate_quantum_hits_parallel(z2, total_photons, n_jobs=-1, target_memory_gb=32.0):
    total_photons = int(total_photons)
    I_classical = compute_classical_intensity(z2).astype(np.float32)
    prob_dist = I_classical / I_classical.sum()
    I_flat = prob_dist.ravel()
    shape = prob_dist.shape

    n_cores = cpu_count() if n_jobs == -1 else n_jobs
    batch_size = estimate_safe_batch_size(target_memory_gb=target_memory_gb)

    print(f"💡 Using {n_cores} CPU cores")
    print(f"📦 Estimated safe batch size: {batch_size:,} photons")
    print(f"🔄 Total batches: {total_photons // batch_size + (total_photons % batch_size > 0)}")

    hit_counts = np.zeros(I_flat.size, dtype=np.int64)
    start_time = time.time()

    num_batches = total_photons // batch_size
    remainder = total_photons % batch_size
    all_batch_sizes = [batch_size] * num_batches
    if remainder > 0:
        all_batch_sizes.append(remainder)

    for i, batch_photons in enumerate(all_batch_sizes, start=1):
        photons_per_core = [batch_photons // n_cores] * n_cores
        for j in range(batch_photons % n_cores):
            photons_per_core[j] += 1

        print(f"🧪 Batch {i}/{len(all_batch_sizes)}: Simulating {batch_photons:,} photons...")
        batch_start = time.time()

        results = Parallel(n_jobs=n_cores)(
            delayed(_sample_photons)(I_flat, n) for n in photons_per_core
        )

        hit_counts += np.sum(results, axis=0)
        batch_time = time.time() - batch_start
        mem_used = psutil.virtual_memory().used / (1024 ** 3)
        print(f"✅ Batch {i} completed in {batch_time:.2f}s | Memory used: {mem_used:.2f} GB")

    total_time = time.time() - start_time
    hit_map = hit_counts.reshape(shape)
    if hit_map.max() > 0:
        hit_map = hit_map / hit_map.max()

    print(f"\n✅ Simulation completed in {total_time:.2f} seconds")
    return hit_map



# --- Initial Settings ---
initial_z2_offset = 3e-6 # TODO (e.g., 1e-3)
initial_photons = 1e6 # TODO (e.g., 1000)
initial_z2 = z1 + initial_z2_offset
extent = (-L/2, L/2, -L/2, L/2)  # for mm scale

# --- Classical Figure (pre-coded reward) ---
# fig1, ax1 = plt.subplots(figsize=(6, 5))
# plt.subplots_adjust(bottom=0.25)
# I_classical = compute_classical_intensity(initial_z2)
# img_classical = ax1.imshow(I_classical, cmap='inferno', extent=extent)
# ax1.set_title("Classical MZI")
# ax1.set_xlabel("x (mm)")
# ax1.set_ylabel("y (mm)")

# ax_slider1 = plt.axes([0.2, 0.1, 0.6, 0.03])
# slider1 = Slider(ax_slider1, 'Path Diff (mm)', 0, 5, valinit=initial_z2_offset * 1e-6)

# def update_classical(val):
#     z2 = z1 + slider1.val * 1e-6
#     img_classical.set_data(compute_classical_intensity(z2))
#     fig1.canvas.draw_idle()

# slider1.on_changed(update_classical)

if __name__ == "__main__":
    # --- Quantum Figure (pre-coded reward) ---
    # fig2, ax2 = plt.subplots(figsize=(6, 5))
    # plt.subplots_adjust(bottom=0.3)
    # hit_map = simulate_quantum_hits_parallel(initial_z2, initial_photons)
    # img_quantum = ax2.imshow(hit_map, cmap='viridis', extent=extent)
    # ax2.set_title("Quantum MZI")
    # ax2.set_xlabel("x (mm)")
    # ax2.set_ylabel("y (mm)")

    # ax_slider2 = plt.axes([0.2, 0.18, 0.6, 0.03])
    # ax_slider3 = plt.axes([0.2, 0.1, 0.6, 0.03])
    # slider2 = Slider(ax_slider2, 'Path Diff (mm)', 0, 5, valinit=initial_z2_offset * 1e-6)
    # slider3 = Slider(ax_slider3, 'Photons', 100, 1e7, valinit=initial_photons, valstep=100)

    # def update_quantum(val):
    #     z2 = z1 + slider2.val * 1e-6
    #     photons = int(slider3.val)
    #     img_quantum.set_data(simulate_quantum_hits_parallel(z2, photons))
    #     fig2.canvas.draw_idle()

    # slider2.on_changed(update_quantum)
    # slider3.on_changed(update_quantum)

    # --- Combined Classical and Quantum Figure (bonus visualization) ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    I_comb = compute_classical_intensity(initial_z2)
    hits_comb = simulate_quantum_hits_parallel(initial_z2, initial_photons)

    img3a = ax3a.imshow(I_comb, cmap='inferno', extent=extent)
    ax3a.set_title("Classical MZI (Combined)")
    ax3a.set_xlabel("x (mm)")
    ax3a.set_ylabel("y (mm)")

    img3b = ax3b.imshow(hits_comb, cmap='viridis', extent=extent)
    ax3b.set_title("Quantum MZI (Combined)")
    ax3b.set_xlabel("x (mm)")
    ax3b.set_ylabel("y (mm)")

    ax_slider4 = plt.axes([0.2, 0.12, 0.6, 0.03])
    ax_slider5 = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider4 = Slider(ax_slider4, 'Path Diff (mm)', 0, 5, valinit=initial_z2_offset * 1e-6)
    slider5 = Slider(ax_slider5, 'Photons', 100, 1e7, valinit=initial_photons, valstep=100)

    def update_combined(val):
        z2 = z1 + slider4.val * 1e-6
        photons = int(slider5.val)
        img3a.set_data(compute_classical_intensity(z2))
        img3b.set_data(simulate_quantum_hits_parallel(z2, photons))
        fig3.canvas.draw_idle()

    slider4.on_changed(update_combined)
    slider5.on_changed(update_combined)

    # --- Show All Plots ---
    plt.show()
