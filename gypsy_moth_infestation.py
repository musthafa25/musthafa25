import ee
import geemap
from gee_image_processing import data_preprocessing
from bug_detect_US import veg_ndmi

# Priori parameters
t1_start_date = '2018-04-01'
t1_end_date = '2022-04-30'
t2_start_date = '2022-07-01'
t2_end_date = '2022-07-30'
mask_year = 2021
CloudCoverMax = 10
gypsy_michi = geemap.shp_to_ee('/Users/aidash/Downloads/Bug_infestation/gypsy_moth_infestation/aoi/Michigan_test_site.shp')
shape_file = gypsy_michi

def read_datasets(start_date, end_date, mask_year, CloudCoverMax, shape_file):
    # Applicable only for US area
    lc_s2 = ee.ImageCollection("USFS/GTAC/LCMS/v2021-7") .filter(ee.Filter.eq('year', mask_year)).first()
    naip_2020 = ee.ImageCollection("USDA/NAIP/DOQQ").filter(ee.Filter.date(start_date, end_date))
    # Ingest the shapefiles - US state
    shp = shape_file

    # Sentinel-2 surface reflectance image
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CloudCoverMax)) \
        .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', CloudCoverMax)) \
        .filterDate(start_date, end_date)\
        .filterBounds(shp)
    print('The Size of the Sentinel-2 image collection is :',s2_sr.size().getInfo())
    print('1. Reading S2 imagecollection, 10 m landuse land cover and shapefile - completed')
    return s2_sr, naip_2020, lc_s2

def save_classified_image(image, shapepath, filename):
    clip_shp = geemap.shp_to_ee(shapepath)
    roi = clip_shp.geometry()
    geemap.ee_export_image(image, filename=filename, scale=10, region=roi, file_per_band=False)

def preprocess_s2(t1_data, t2_data):
    s2_sr_t1_mean = data_preprocessing(s2_sr_t1, lc_s2)
    s2_sr_t2_mean = data_preprocessing(s2_sr_t2, lc_s2)
    return s2_sr_t1_mean, s2_sr_t2_mean

def calculate_ndmi(t1_data, t2_data):
    t1_ndmi = veg_ndmi(s2_sr_t1_mean).add(1)
    t2_ndmi = veg_ndmi(s2_sr_t2_mean).add(1)
    print('3. NDMI calculated for t1 and t2 imagery; added 1 for rescaling the products')
    return t1_ndmi, t2_ndmi

def gypsy_detect(t1_ndmi, t2_ndmi):
    q1 = t1_ndmi.divide(2).subtract(t2_ndmi.divide(2))
    c_map = t1_ndmi.multiply(0).where(q1.gte(0.05), 1)
    c_map = c_map.where(q1.lt(0.05), 0)
    return c_map

def save_classified_image(image, shapefile, filename):
    roi = shape_file.geometry()
    geemap.ee_export_image(image, filename=filename, scale=100, region=roi, file_per_band=False)

if __name__ == '__main__':
    s2_sr_t1, naip_2020, lc_s2 = read_datasets(t1_start_date, t1_end_date, mask_year, CloudCoverMax, shape_file)
    s2_sr_t2, naip_2020, lc_s2 = read_datasets(t2_start_date, t2_end_date, mask_year, CloudCoverMax, shape_file)
    s2_sr_t1_mean, s2_sr_t2_mean = preprocess_s2(s2_sr_t1, s2_sr_t2)
    t1_ndmi, t2_ndmi = calculate_ndmi(s2_sr_t1_mean, s2_sr_t2_mean)
    c_map = gypsy_detect(t1_ndmi, t2_ndmi)
    save_classified_image(c_map, shape_file, filename='gypsy_moth_2022.tif')







