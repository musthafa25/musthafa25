import ee
import geemap
from geemap import ml
import geopandas as gpd
from gee_image_processing import read_datasets
from gee_image_processing import data_preprocessing
from gee_image_processing import calculate_indices


# 1. Priori parameters
start_date = '2021-01-01'
end_date = '2022-12-31'
mask_year = 2021
CloudCoverMax = 10

# Shapefiles
us_states = gpd.read_file('data/US_shapefiles/us-states.shp')
shape_file = us_states[us_states['name'] == 'Arkansas'] # Select appropriate state for analysis

def change_bandnames(image, indices_list):
    lis = []
    for i in range(len(indices_list)):
        x = indices_list[i][:-4]
        lis += [x]
    inBands = image.bandNames().getInfo()
    outBands = lis
    image = image.select(inBands, outBands)
    return image

def readFile(fileName):
    fileObj = open(fileName, "r")  # opens the file in read mode
    words = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    return words

def save_classified_image(image, shapepath, filename):
    clip_shp = geemap.shp_to_ee(shapepath)
    roi = clip_shp.geometry()
    geemap.ee_export_image(
        image, filename=filename, scale=10, region=roi, file_per_band=False
    )

if __name__ == '__main__':
    s2_sr, naip_2020, lc_s2 = read_datasets(start_date, end_date, mask_year, CloudCoverMax, shape_file=shape_file)
    s2_sr_mean = data_preprocessing(s2_sr, lc_s2)
    indices_collection, indices_list = calculate_indices(s2_sr_mean)
    img_indices = ee.ImageCollection.toBands(indices_collection) # converting list of indices into a multiband image
    img_indices = change_bandnames(img_indices, indices_list) # Change band names of the imagery
    band_names = readFile('features.txt') # Select the features as per the model developed
    image = img_indices.select(band_names)
    classifier = ml.csv_to_classifier('data/bug_ml_model.csv')
    classified_img = image.select(band_names).classify(classifier)
    # Save the file when required
    shape_path = 'data/shapefiles/vhr_image_outline.shp'
    file_name = 'us_arkansas.tif'
    save_classified_image(classified_img, shapepath=shape_path, filename='us_arkansas_10m.tif')













