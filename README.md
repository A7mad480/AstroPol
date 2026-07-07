AstroPol Pipeline
An automated Python pipeline for astronomical image processing, star tracking, aperture photometry, and polarimetry analysis implemented in run_pipeline.py. It processes FITS images, extracts precise centroids, and computes Stokes parameters and normalized differences for E-beam and O-beam optical beams.

Features
Photometry & Tracking: Automatically tracks stars, computes precise centroids, and performs aperture photometry across sequential image frames.

Polarimetry Analysis: Fits single-beam intensity data to a Stokes model and calculates true net polarization using a normalized difference method (E-O)/(E+O).

Visualization: Generates multi-panel diagnostic plots displaying measured fluxes, mathematical fits, and net polarization curves.

Prerequisites
Ensure you have the required Python packages installed in your environment:

Plaintext
numpy, pandas, matplotlib, astropy, photutils, scipy


Usage
Download or clone this repository to your local machine.

Open run_pipeline.py and update the local file and folder paths in the main execution block to match your directory structure:

Python
data_folder = r"C:\Path\To\Your\AstroPol_Test\data" 
csv_output_path = r"C:\Path\To\Your\AstroPol_Test\flux_results.csv"
plot_output_path = r"C:\Path\To\Your\AstroPol_Test\Polarization_Results.png"
Run the script from your terminal or command prompt:

Bash
python run_pipeline.py

License
Distributed under the MIT License. See the LICENSE file for more details.
