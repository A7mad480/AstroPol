import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit

# ==========================================
# 1. Photometry & Tracking Functions
# ==========================================

def get_angle(file_path, index, method, step_deg, header_keyword):
    """Extract the retarder angle based on the chosen method."""
    if method == 'sequential':
        return index * step_deg
    elif method == 'header':
        header = fits.getheader(file_path)
        return header.get(header_keyword, 0.0)
    elif method == 'filename':
        match = re.search(r'\d+\.?\d*', file_path.split('/')[-1].split('\\')[-1])
        return float(match.group()) if match else 0.0
    else:
        raise ValueError("Method must be 'sequential', 'header', or 'filename'")

def manual_star_selection(image_data):
    """Display the first image to manually select stars once."""
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title("Click on E-beam FIRST, then O-beam SECOND. Press Enter when done.")
    
    coords = plt.ginput(2, timeout=0)
    plt.close(fig)
    
    if len(coords) < 2:
        raise Exception("You must select exactly two stars (E-beam and O-beam).")
    
    print(f"Selected coordinates -> E-beam: {coords[0]}, O-beam: {coords[1]}")
    return coords[0], coords[1]

def process_image_folder(folder_path, output_csv, aperture_radius=25, box_size=40, 
                         angle_method='sequential', step_deg=22.5, header_keyword='HWP_ANGL'):
    """Read images, track stars, and compute flux."""
    image_files = sorted(glob.glob(f"{folder_path}/*.fit*"))
    if not image_files:
        raise FileNotFoundError(f"No FITS files found in the folder: {folder_path}")
    
    print(f"Found {len(image_files)} images. Opening the first image for selection...")
    
    first_image_data = fits.getdata(image_files[0])
    current_e_pos, current_o_pos = manual_star_selection(first_image_data)
    
    results = []
    
    for i, file_path in enumerate(image_files):
        data = fits.getdata(file_path)
        angle = get_angle(file_path, i, angle_method, step_deg, header_keyword)
        
        fluxes = []
        for beam_name, pos in [("E-beam", current_e_pos), ("O-beam", current_o_pos)]:
            x, y = int(pos[0]), int(pos[1])
            
            # Cut out a box to track the center
            y_min, y_max = max(0, y - box_size//2), min(data.shape[0], y + box_size//2)
            x_min, x_max = max(0, x - box_size//2), min(data.shape[1], x + box_size//2)
            cutout = data[y_min:y_max, x_min:x_max]
            
            # Compute precise centroid
            xc, yc = centroid_com(cutout)
            exact_x = x_min + xc
            exact_y = y_min + yc
            
            # Compute flux
            aperture = CircularAperture((exact_x, exact_y), r=aperture_radius)
            phot_table = aperture_photometry(data, aperture)
            flux = phot_table['aperture_sum'][0]
            fluxes.append(flux)
            
            # Update coordinates for the next image
            if beam_name == "E-beam":
                current_e_pos = (exact_x, exact_y)
            else:
                current_o_pos = (exact_x, exact_y)
                
        results.append({
            'frame': i + 1,
            'e_beam_flux': fluxes[0],
            'o_beam_flux': fluxes[1],
            'angle_deg': angle
        })
        print(f"Processed frame {i+1}/{len(image_files)} | Angle: {angle}°")
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Photometry complete. Data saved to {output_csv}\n")
    return output_csv

# ==========================================
# 2. Polarimetry & Plotting Functions
# ==========================================

def stokes_model(theta, I, Q, U):
    """Stokes model for single beam intensity."""
    return 0.5 * I + 0.5 * Q * np.cos(4 * theta) + 0.5 * U * np.sin(4 * theta)

def calculate_polarization(angles_deg, flux_values):
    """Calculate polarization parameters for single beams E and O."""
    theta = np.deg2rad(angles_deg)
    p0 = [2 * np.mean(flux_values), 0, 0]
    popt, _ = curve_fit(stokes_model, theta, flux_values, p0=p0)
    
    I, Q, U = popt
    p = np.sqrt(Q**2 + U**2) / I
    chi = 0.5 * np.rad2deg(np.arctan2(U, Q))
    chi = chi % 180 
    
    return {'I': I, 'Q': Q, 'U': U, 'p': p, 'chi': chi}

def analyze_and_plot(csv_file_path, output_plot):
    """Precise scientific function to calculate and plot polarization using Normalized Difference."""
    df = pd.read_csv(csv_file_path)
    angles = df['angle_deg'].to_numpy()
    e_flux = df['e_beam_flux'].to_numpy()
    o_flux = df['o_beam_flux'].to_numpy()
    
    # 1. Calculate polarization for individual beams
    e_params = calculate_polarization(angles, e_flux)
    o_params = calculate_polarization(angles, o_flux)
    
    # 2. Academic calculation of the normalized difference
    norm_diff = (e_flux - o_flux) / (e_flux + o_flux)
    
    theta = np.deg2rad(angles)
    def norm_model(th, c, Q, U):
        return c + Q * np.cos(4 * th) + U * np.sin(4 * th)
        
    p0 = [np.mean(norm_diff), 0, 0]
    popt, _ = curve_fit(norm_model, theta, norm_diff, p0=p0)
    c, Q, U = popt
    
    p_true = np.sqrt(Q**2 + U**2)
    chi_true = (0.5 * np.rad2deg(np.arctan2(U, Q))) % 180
    
    # 3. Setup the plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Polarimetry Analysis: E-beam, O-beam, and Normalized Difference', fontsize=16)
    
    theta_smooth = np.linspace(0, max(angles), 200)
    theta_smooth_rad = np.deg2rad(theta_smooth)
    
    # Plot E-beam
    axes[0].plot(angles, e_flux, 'bo', label='Measured E-beam')
    axes[0].plot(theta_smooth, stokes_model(theta_smooth_rad, e_params['I'], e_params['Q'], e_params['U']), 'b-', 
                 label=f"Fit: p={e_params['p']*100:.2f}%, chi={e_params['chi']:.1f}°")
    axes[0].set_ylabel('E-beam Flux')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot O-beam
    axes[1].plot(angles, o_flux, 'ro', label='Measured O-beam')
    axes[1].plot(theta_smooth, stokes_model(theta_smooth_rad, o_params['I'], o_params['Q'], o_params['U']), 'r-', 
                 label=f"Fit: p={o_params['p']*100:.2f}%, chi={o_params['chi']:.1f}°")
    axes[1].set_ylabel('O-beam Flux')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Normalized Difference (true net polarization)
    axes[2].plot(angles, norm_diff, 'go', label='Normalized Difference (E-O)/(E+O)')
    axes[2].plot(theta_smooth, norm_model(theta_smooth_rad, c, Q, U), 'g-', 
                 label=f"True Fit: p={p_true*100:.3f}%, chi={chi_true:.1f}°")
    axes[2].set_xlabel('HWP Angle (Degrees)')
    axes[2].set_ylabel('Normalized Diff Flux')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save plot
    plt.savefig(output_plot, dpi=300)
    print(f"Polarimetry complete! Plot saved as {output_plot}")
    plt.show()

# ==========================================
# 3. Main Execution Block
# ==========================================

if __name__ == "__main__":
    # File and folder paths (update to match your local directory to avoid path issues)
    # Use raw string r"" to avoid Windows path escape issues
    data_folder = r"C:\Users\ahmad\OneDrive\Desktop\AstroPol_Test\data" 
    csv_output_path = r"C:\Users\ahmad\OneDrive\Desktop\AstroPol_Test\flux_results.csv"
    plot_output_path = r"C:\Users\ahmad\OneDrive\Desktop\AstroPol_Test\Polarization_Results.png"
    
    # Run image processing and flux extraction
    csv_result = process_image_folder(
        folder_path=data_folder, 
        output_csv=csv_output_path,
        angle_method='sequential', 
        step_deg=22.5,
        aperture_radius=25 
    )
    
    # Run physical calculations and plotting
    analyze_and_plot(csv_result, output_plot=plot_output_path)