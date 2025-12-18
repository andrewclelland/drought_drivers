"""
Baseline script to process images from ERA5-Land to locally-saved NumPy arrays using Google Earth Engine.
"""
import ee
import numpy as np
import time
from datetime import date, timedelta

# Record the start time (bonus step if interested in how long things take to run)
start_time = time.time()

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='spherical-berm-323321') # <-- Edit project name to the one which your account is linked

# Import grid to iterate over and any region shapefile for clipping
# A grid is super useful as the code will iterate over each cell in turn, which reduces the GEE memory usage (which is limited)
final_grid = ee.FeatureCollection('users/andyc97/model_shapefiles/final_grid')
final_shp = ee.FeatureCollection('users/andyc97/model_shapefiles/final_north')

# Example for loading bands/images from assets
# Eg these could be added as bands to an image, but here it's used for masking
base_land = ee.Image('users/andyc97/model_shapefiles/final_baseland')
aspect = base_land.select('aspect')

# Consistent Parameters
years = range(2015, 2025) # 2015 to 2024
months = range(1, 13) # 1 to 12
days = range(1, 32) # 1 to 31

# Select band names for processing, same as the names within the GEE ERA5-Land catalog
# Include parameters which will be needed to process other variables (eg for wind speed and relative humidity)
selected_bands = ['total_precipitation_sum', 'temperature_2m', 'temperature_2m_max', 'temperature_2m_min', 'dewpoint_temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m']

# --- Function to generate an ee.Image from the ERA5-Land collection in GEE ---
def create_image(year, month, day, selected_bands):
    # Loop over each day using timedelta
    # Even though we only consider one day in turn, the end_date is the following day due to standard Python reasoning
    start = date(year, month, day)
    end = start + timedelta(days=1)
    start_date = start.isoformat()
    end_date = end.isoformat()

    # Load in each ERA5-Land image for a day
    # This will initially be as an ImageCollection with 1 image - call using .first(). It's stupid logic but welcome to Earth Engine.
    # If interested in other available datasets (eg SMAP) change the ee.ImageCollection root as per the catalog and go from there
    era5land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date)
    image = era5land.first()

    # Reproject image
    image = image.reproject(crs='EPSG:4326', scale=4000) # <-- Edit CRS and projection scale (in metres) as necessary
    
    # Select specific bands
    image = image.select(selected_bands) # This means GEE will only consider the relevant bands rather than all 150

    # Function for calculating relative humidity
    # Different functions for rh are available which will affect the constant values. This one is based on the Magnus formula (1844)
    def calculate_rh(image):
        temp = image.select('temperature_2m').subtract(273.15) # Temperature in ERA5-Land is in Kelvin by default
        dewpoint = image.select('dewpoint_temperature_2m').subtract(273.15)
        # Calculate saturation vapor pressure for temperature and dewpoint, then rh
        e_t = image.expression('6.112 * exp((17.67 * temp) / (temp + 243.5))', {'temp': temp})
        e_td = image.expression('6.112 * exp((17.67 * dewpoint) / (dewpoint + 243.5))', {'dewpoint': dewpoint})
        rh = e_td.divide(e_t).multiply(100).clamp(0, 100)  # Clamp to range [0, 100]
        return rh
    
    # Function for calculating wind speed
    def calculate_wsp(image):
        u10 = image.select('u_component_of_wind_10m')
        v10 = image.select('v_component_of_wind_10m')
        wspe5l = (u10.pow(2).add(v10.pow(2))).sqrt()
        return wspe5l

    # The multiply-round-divide steps are included to reduce the number of dp (and thus memory), but can be removed
    # .map is only available for an ImageCollection - hence we call era5land.map then .first() (because of Earth Engine logic)
    hurs = era5land.map(calculate_rh).first().multiply(100).round().divide(100).rename("relative_humidity")
    wsp = era5land.map(calculate_wsp).first().multiply(100).round().divide(100).rename("wind_speed")

    # Add these newly-calculated bands to the image
    # Any other bands from other images (eg static images from EE assets) can also be added here
    image = image.addBands(hurs).addBands(wsp)

    # Fill missing data with -9999, clip and update the mask to another image 
    # Clipping and updating the mask is in effect a double-double check, 
    # not strictly necessary but we want all the arrays to be the same length otherwise we got big issues
    image = image.unmask(-9999, sameFootprint=True).clip(final_shp).updateMask(aspect)
    
    # Additional step for saving as GeoTIFF file only
    image = image.set({
        'year': year,
        'month': month,
        'day': day,
        'system:time_start': ee.Date(start_date).millis(),
        'system:time_end': ee.Date(end_date).millis()
    })
    return image

# Choosen final bands for processing to arrays - ignore additional bands
final_bands = ['total_precipitation_sum', 'temperature_2m', 'temperature_2m_max', 'temperature_2m_min', 'relative_humidity', 'wind_speed']

# --- Process each band in the image to an individual array ---
for year in years:
    for month in months:
        for day in days:
            try:
                image = create_image(year, month, day, selected_bands)
                extracted_data = final_grid.map(lambda cell: cell.set(image.reduceRegion( # This is where the grid is useful to iterate through the cells
                    reducer=ee.Reducer.toList(), # different Reducers are available for different tasks. Here is .toList()
                    geometry=cell.geometry(),
                    scale=4000, # <-- Edit as necessary
                    bestEffort=False, # bestEffort=False ensures that pixels at the edge of the cell aren't rounded
                    crs='EPSG:4326', # <-- Edit as necessary
                    maxPixels=1e9
                )))

                # 'features' is the GEE way of calling each band
                # .getInfo() is a common term for viewing/loading the separate parts of each GEE-formatted file
                features = extracted_data.getInfo()['features']
                concatenated_data = {band: [] for band in final_bands}
                
                # Process each band to array, converting the -9999 back to NaN (this is necessary for tree-based models)
                for feature in features:
                    for band in final_bands:
                        data = np.array(feature['properties'].get(band, []))
                        data = np.where(data == -9999, np.nan, data)
                        concatenated_data[band].extend(data)

                # Reshape the array into the correct shape and save for each individual band
                for band in final_bands:
                    final_band_array = np.array(concatenated_data[band]).reshape(-1, 1)
                    filename = f'/rds/general/user/aac115/home/drought_drivers/processed_{year}_{month:02d}_{day:02d}_{band}_array.npy' # <-- Edit as necessary
                    np.save(filename, final_band_array)
                    print(f"Saved: {filename} with shape {final_band_array.shape}")
                    
            except Exception as e:
                # Erroneous dates, such as "30th Feb" will appear as errors but the code will automatically continue. This is a good sanity check
                print(f"Error processing {year}-{month}-{day}: {e}") 

# Calculate the time difference (bonus step as with the start time)
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600  # Convert seconds to hours
print(f"Time taken: {time_difference_hours:.2f} hours")