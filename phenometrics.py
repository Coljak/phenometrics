"""
This script is for the cluster!
This script takes a folder path of NDVI tifs  as input.
The NDVI tifs are processed pixelwise to determine the start of season (sos) and end of season (eos) of the vegetation.

This Version cuts off all values before and after the minimums of Start of Season and End of Season.
Base, Amplitude and seasonal_amplitude are not revised yet.
"""

import os
import numpy as np
# import pandas as pd
import concurrent.futures
import rasterio
from scipy.interpolate import splrep, BSpline
from sklearn.linear_model import LinearRegression
from datetime import datetime
# from rasterio.crs import CRS
# from affine import Affine

# Constants
SMOOTH = 0.05
THRESHOLD_ = 90 # can be set to None
THRESH_END = 250
MAX_WORKERS = 50
CHUNK_SIZE = 1000

def load_ndvi_stack(directory, filter_threshold=THRESHOLD_):
    """
    Load NDVI stack from a directory containing NDVI images.
    If not set manually the threshold is set from the constants to an int between 0 and 365.
    A two dimensional mask is returned that is true for all pixels that hold at least one value 
    """
    print('Loading NDVI stack from', directory)
    # TODO: shouldn't the first dataset below the threshold be taken into consideration?

    file_paths = sorted([os.path.join(directory, fp) for fp in os.listdir(directory) if fp.endswith('.tif')])
    
    def sort_by_doy(file_paths):
        fps = np.array(file_paths)
        fp_doys = np.array([os.path.basename(fp).split('_')[-2].split('.')[0] for fp in fps])
        sorted_indices = np.argsort(fp_doys)
        return fps[sorted_indices]
    
    file_paths = sort_by_doy(file_paths)
    
    doys = np.array([int(os.path.basename(fp).split('_')[-2].split('.')[0]) for fp in file_paths], dtype=np.int64)
    
    if filter_threshold is not None:
        file_paths = [fp for fp in file_paths if int(os.path.basename(fp).split('_')[-2].split('.')[0]) >= filter_threshold]
        doys = np.array([int(os.path.basename(fp).split('_')[-2].split('.')[0]) for fp in file_paths], dtype=np.int64)
          
    ndvi_stack = []
    mask_stack = []
       
    for fp in file_paths:
        with rasterio.open(fp) as dataset:
            ndvi_data = dataset.read(1).astype(np.float32)
            # ndvi_stack.append(ndvi_data[:300, :300])
            ndvi_stack.append(ndvi_data)
            
            mask_stack.append(np.isnan(ndvi_data))
    with rasterio.open(file_paths[0]) as dataset:
        meta = dataset.meta

    ndvi_stack = np.array(ndvi_stack)
    mask_stack = np.array(mask_stack)

    valid_image_area_mask = ~np.any(mask_stack, axis=0)
    if len(doys) < 3:
        raise ValueError(f'Not enough images in the stack (less than 3!) in directory {directory} with threshold {filter_threshold}')
    
    return ndvi_stack, doys, valid_image_area_mask, meta

def find_max(doys, ndvi_values, THRESH_END):
    mask = doys <= THRESH_END
    filtered_doys = doys[mask]
    filtered_ndvi_values = ndvi_values[mask]
    max_index = np.argmax(filtered_ndvi_values)
    return filtered_doys[max_index], filtered_ndvi_values[max_index]

def filter_to_range_between_minima(fine_doys, b_spline, max_ndvi_doy):
    mask_sos = fine_doys < max_ndvi_doy
    mask_eos = fine_doys > max_ndvi_doy
    
    filtered_doys_sos = fine_doys[mask_sos]
    filtered_b_spline_sos = b_spline[mask_sos]
    
    filtered_doys_eos = fine_doys[mask_eos]
    filtered_b_spline_eos = b_spline[mask_eos]
    
    # indices of minimum NDVI values on both sides of max_ndvi_doy
    index_sos = np.argmin(filtered_b_spline_sos)
    index_eos = np.argmin(filtered_b_spline_eos)
    
    # Convert these indices back to the original indices
    original_index_sos = np.where(fine_doys == filtered_doys_sos[index_sos])[0][0]
    original_index_eos = np.where(fine_doys == filtered_doys_eos[index_eos])[0][0]
    
    # subsets of fine_doys and b_spline between the two NDVI minima
    b_spline_new = b_spline[original_index_sos:(original_index_eos+1)]
    fine_doys_new = fine_doys[original_index_sos:(original_index_eos+1)]
    
    
    return fine_doys_new, b_spline_new

def fit_linear_regression(fine_doys, b_spline, max_ndvi_doy):
    left_indices = fine_doys < max_ndvi_doy
    right_indices = fine_doys > max_ndvi_doy

    left_model = LinearRegression().fit(fine_doys[left_indices].reshape(-1, 1), b_spline[left_indices]) if np.any(left_indices) else None
    right_model = LinearRegression().fit(fine_doys[right_indices].reshape(-1, 1), b_spline[right_indices]) if np.any(right_indices) else None
    
    return left_model, right_model

def calculate_base(left_model, right_model):
    if left_model is not None and right_model is not None:
        return (left_model.coef_[0] + right_model.coef_[0]) / 2.0
    else:
        return np.nan
    
def calculate_relative_amplitude(ndvi_values):
    return np.percentile(ndvi_values, 90) - np.percentile(ndvi_values, 10)


def calculate_sos_eos(fine_doys, b_spline, base, amplitude, max_ndvi_doy, max_ndvi_value, overall_relative_amplitude):
    """
    Calculations are split into sos - start of season and eos - end of season.
    Those timeranges are defined by the maximum NDVI value in the seaseon.
    """
# def calculate_sos_eos(fine_doys, b_spline, base, amplitude, max_ndvi_doy):    
    # b_spline = BSpline(*tck)(fine_doys)
    sos_mask = fine_doys < max_ndvi_doy
    eos_mask = fine_doys > max_ndvi_doy
    sos_doys = fine_doys[sos_mask]
    eos_doys = fine_doys[eos_mask]
    sos_ndvi_values = b_spline[sos_mask]
    eos_ndvi_values = b_spline[eos_mask]

    seasonal_amplitude = base + 0.25 * amplitude 
    
    # SoS calculations
    if len(sos_ndvi_values) > 0:
        sos_min_value = np.min(sos_ndvi_values)
        threshold_sos = sos_min_value + 0.1 * (max_ndvi_value - sos_min_value)
        #TODO Genaugenommen müsste hier der np.abs wegfallen, weil ABOVE thrs
        # JETZT ERSETZT DURCH first_of_slope10_sos2
        # first_of_slope10_sos = sos_doys[np.argmin(np.abs(sos_ndvi_values - threshold_sos))]
        
        sos_diff = sos_ndvi_values - threshold_sos
        positive_sos_diff = sos_diff[sos_diff > 0]
        if len(positive_sos_diff) > 0:
        # TODO delete this or first_of-slope10_sos
            first_of_slope10_sos2 = sos_doys[sos_diff > 0][np.argmin(sos_diff[sos_diff > 0])]
        else:
            first_of_slope10_sos2 = np.nan
            print(f'Pixel first_of_slope10_sos2 is nan')
        
        median_ndvi_sos = np.median(sos_ndvi_values)
        median_of_slope_sos = sos_doys[np.argmin(np.abs(sos_ndvi_values - median_ndvi_sos))] 
         
        seasonal_amplitude_doy_sos = sos_doys[np.argmin(np.abs(sos_ndvi_values - seasonal_amplitude))]
        
        relative_amplitude_sos_old = calculate_relative_amplitude(sos_ndvi_values)
        relative_amplitude_doy_sos_old_idx = np.argmin(np.abs(sos_ndvi_values - relative_amplitude_sos_old))
        relative_amplitude_doy_sos_old = sos_doys[relative_amplitude_doy_sos_old_idx]
        if relative_amplitude_doy_sos_old_idx == 0:
            relative_amplitude_doy_sos_old = np.nan


        # relative to the overall amplitude
        
        dists_from_relative_amplitude = abs(sos_ndvi_values - overall_relative_amplitude)
        idx_relative_amplitude = np.argmin(dists_from_relative_amplitude)
        if idx_relative_amplitude > 0:
            relative_amplitude_doy_sos = sos_doys[idx_relative_amplitude]
        else:
            relative_amplitude_doy_sos = np.nan

    else:
        # first_of_slope10_sos = np.nan
        first_of_slope10_sos2 = np.nan
        median_of_slope_sos = np.nan
        seasonal_amplitude_doy_sos = np.nan
        relative_amplitude_doy_sos = np.nan
        relative_amplitude_doy_sos_old = np.nan

    # EoS calculations
    if len(eos_ndvi_values) > 0:
        eos_min_value = np.min(eos_ndvi_values)
        threshold_eos = eos_min_value + 0.1 * (max_ndvi_value - eos_min_value)
        # Siehe oben!
        # first_of_slope10_eos = eos_doys[np.argmin(np.abs(eos_ndvi_values - threshold_eos))]

        eos_diff = eos_ndvi_values - threshold_eos
        positive_eos_diff = eos_diff[eos_diff < 0]
        if len(positive_eos_diff) > 0:
        # TODO delete this or first_of-slope10_sos
            first_of_slope10_eos2 = eos_doys[eos_diff < 0][np.argmax(eos_diff[eos_diff < 0])]
        else:
            first_of_slope10_eos2 = np.nan
            print(f'Pixel first_of_slope10_eos2 is nan')
        
        median_ndvi_eos = np.median(eos_ndvi_values)
        median_of_slope_eos = eos_doys[np.argmin(np.abs(eos_ndvi_values - median_ndvi_eos))]
        
        
        seasonal_amplitude_doy_eos = eos_doys[np.argmin(np.abs(eos_ndvi_values - seasonal_amplitude))]
        
        # old relative amplitude
        relative_amplitude_eos_old = calculate_relative_amplitude(eos_ndvi_values)
        relative_amplitude_eos_old_idx = np.argmin(np.abs(eos_ndvi_values - relative_amplitude_eos_old))
        relative_amplitude_doy_eos_old = eos_doys[relative_amplitude_eos_old_idx]
        if relative_amplitude_eos_old_idx == 0:
            relative_amplitude_doy_eos_old = np.nan

        # relative to the overall amplitude
        dists_from_relative_amplitude = abs(eos_ndvi_values - overall_relative_amplitude)  
        idx_relative_amplitude = np.argmin(dists_from_relative_amplitude)
        if idx_relative_amplitude > 0:
            relative_amplitude_doy_eos = eos_doys[idx_relative_amplitude]
        else:
            relative_amplitude_doy_eos = np.nan
    else:
        first_of_slope10_eos2 = np.nan
        median_of_slope_eos = np.nan
        seasonal_amplitude_doy_eos = np.nan
        relative_amplitude_doy_eos = np.nan
        relative_amplitude_doy_eos_old = np.nan
    
    return {
        'sos_first_of_slope': first_of_slope10_sos2, 
        'sos_median_of_slope': median_of_slope_sos, 
        'sos_seasonal_amplitude': seasonal_amplitude_doy_sos, 
        'sos_relative_amplitude': relative_amplitude_doy_sos,
        'sos_relative_amplitude_old': relative_amplitude_doy_sos_old,
        'eos_first_of_slope': first_of_slope10_eos2,
        'eos_median_of_slope': median_of_slope_eos,
        'eos_seasonal_amplitude': seasonal_amplitude_doy_eos,
        'eos_relative_amplitude': relative_amplitude_doy_eos,
        'eos_relative_amplitude_old': relative_amplitude_doy_eos_old,
    }


def process_pixel(row, col, ndvi_values, doys, num_cols):
    """
    1. Nan values in the pixel are removed
    2. Spline function is determined
    3. the overall relative amplitude is determined
    4. The result for one pixel is completed with the values from sos_eos_calculate    
    """
    print('process_pixel ', row, col)

    # if there are invalid values, they are taken out of the ndvi_values and doys
    isnan_mask = np.isnan(ndvi_values)
    if np.any(isnan_mask):
        ndvi_values = ndvi_values[~isnan_mask]
        doys = doys[~isnan_mask]

    # create an array with all doys in the timerange
    min_doy, max_doy = min(doys), max(doys)
    fine_doys = np.linspace(min_doy, max_doy, max_doy - min_doy + 1, dtype=int)

    tck_spline = splrep(doys, ndvi_values, s=SMOOTH)
    b_spline = BSpline(*tck_spline)(fine_doys)

    max_ndvi_doy, max_ndvi_value = find_max(fine_doys, b_spline, THRESH_END)
    left_model, right_model = fit_linear_regression(fine_doys, b_spline, max_ndvi_doy)
    base = calculate_base(left_model, right_model)
    amplitude = max_ndvi_value - base

    overall_relative_amplitude = calculate_relative_amplitude(ndvi_values)

    sos_eos_dict = calculate_sos_eos(fine_doys,  b_spline, base, amplitude, max_ndvi_doy, max_ndvi_value, overall_relative_amplitude)

    sos_eos_dict.update({'pixel_idx': row * num_cols + col, 'max_ndvi_doy': max_ndvi_doy, 'max_ndvi_value': max_ndvi_value})

    return sos_eos_dict


def process_stack(input_dir):
    """
    Main function to process NDVI data and to write the results into the np.array sos_eos_data.
    Once finished, the result tifs are written into a folder on the same level as the input images.
    """
    start = datetime.now()

    print(f'Start processing {input_dir}')
    ndvi_stack, doys, mask, meta = load_ndvi_stack(input_dir)
    num_rows, num_cols = ndvi_stack[0].shape
    print(input_dir, ndvi_stack[0].shape)
    # number of keys == number of result parameters

    num_keys = 12
    sos_eos_data = np.full((num_rows, num_cols, num_keys),np.nan, dtype=np.float32)


    
    for row in range(num_rows):
        for col in range(num_cols):
            if mask[row, col]:
                
                sos_eos_dict = process_pixel(row, col, ndvi_stack[:, row, col], doys, num_cols)

                sos_eos_data[row, col, 0] = sos_eos_dict['max_ndvi_doy']
                sos_eos_data[row, col, 1] = sos_eos_dict['max_ndvi_value']
                sos_eos_data[row, col, 2] = sos_eos_dict['sos_first_of_slope']
                sos_eos_data[row, col, 3] = sos_eos_dict['sos_median_of_slope']
                sos_eos_data[row, col, 4] = sos_eos_dict['sos_seasonal_amplitude']
                sos_eos_data[row, col, 5] = sos_eos_dict['sos_relative_amplitude']
                sos_eos_data[row, col, 6] = sos_eos_dict['sos_relative_amplitude_old']
                sos_eos_data[row, col, 7] = sos_eos_dict['eos_first_of_slope']
                sos_eos_data[row, col, 8] = sos_eos_dict['eos_median_of_slope']
                sos_eos_data[row, col, 9] = sos_eos_dict['eos_seasonal_amplitude']
                sos_eos_data[row, col, 10] = sos_eos_dict['eos_relative_amplitude']
                sos_eos_data[row, col, 11] = sos_eos_dict['eos_relative_amplitude_old']

    
    # An output folder is created at the same level as the input_dir. The output folder is named e.g. 2024-11-06_output.
    current_date = datetime.now().date().isoformat()

    def make_directory(input_dir, counter=1):
        output_dir = os.path.join(os.path.dirname(os.path.dirname(input_dir)), f'{current_date}_output_v{counter}')
        if os.path.exists(output_dir):
            return make_directory(input_dir, counter + 1)
        else:
            os.makedirs(output_dir)
        return output_dir

    output_dir = make_directory(input_dir)

    save_geotiff(sos_eos_data[:, :, 0], meta, os.path.join(output_dir, 'max_ndvi_doy.tif'))
    save_geotiff(sos_eos_data[:, :, 1], meta, os.path.join(output_dir, 'max_ndvi_value.tif'))
    save_geotiff(sos_eos_data[:, :, 2], meta, os.path.join(output_dir, 'sos_first_of_slope.tif'))
    save_geotiff(sos_eos_data[:, :, 3], meta, os.path.join(output_dir, 'sos_median_of_slope.tif'))
    save_geotiff(sos_eos_data[:, :, 4], meta, os.path.join(output_dir, 'sos_seasonal_amplitude.tif'))
    save_geotiff(sos_eos_data[:, :, 5], meta, os.path.join(output_dir, 'sos_relative_amplitude.tif'))
    save_geotiff(sos_eos_data[:, :, 6], meta, os.path.join(output_dir, 'sos_relative_amplitude_old.tif'))
    save_geotiff(sos_eos_data[:, :, 7], meta, os.path.join(output_dir, 'eos_first_of_slope.tif'))
    save_geotiff(sos_eos_data[:, :, 8], meta, os.path.join(output_dir, 'eos_median_of_slope.tif'))
    save_geotiff(sos_eos_data[:, :, 9], meta, os.path.join(output_dir, 'eos_seasonal_amplitude.tif'))
    save_geotiff(sos_eos_data[:, :, 10], meta, os.path.join(output_dir, 'eos_relative_amplitude.tif'))
    save_geotiff(sos_eos_data[:, :, 11], meta, os.path.join(output_dir, 'eos_relative_amplitude_old.tif'))


    print(f'Processing {input_dir} took', datetime.now() - start)

    return f"Processing of the directory {input_dir} was completed in {datetime.now() - start}."


def find_folder_paths(folder_path, paths_list=None):
    """
    A path_list is returned, that contains all folder paths holding tif images within the input folder path.
    """

    if paths_list is None:
        paths_list = []
    
    if any(f.endswith('.tif') for f in os.listdir(folder_path)):
    # if any(f.endswith('.tif') for f in os.listdir(folder_path)) and 'clip' in folder_path.split(os.path.sep):
        if 'output' not in folder_path.split('_'):
            paths_list.append(folder_path)
    else:
        # Check if the input path contains folders
        subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        
        for subfolder in subfolders:
            find_folder_paths(subfolder, paths_list)
    
    return paths_list



def save_geotiff(data, meta, file_path):
    """
    The metadata of the saved tiff is taken from an input tif.
    """
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)


def main(input_dir):
    """
    every NDVI stack will be processes on a seperate thread.
    """
    path_list = []
    try:
        path_list = find_folder_paths(input_dir)
    except:
        print(f'Error: no valid path found in {input_dir} (/clip needs to be the folder containing the tifs).')
        return
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_stack, path) for path in path_list]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  
                print(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
            

if __name__ == '__main__':
    """
    Replace input _dir with the relative or absolute path to either one folder with tif images OR the top level folder that contains all folders with .tif images.
    Only folders containing valid images will be processed.
    """
    input_dir = './S2_clip/'

    start = datetime.now()
    main(input_dir)
    print(f'Time taken for directory {input_dir}: {datetime.now() - start}.')

