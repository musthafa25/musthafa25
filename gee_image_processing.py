# Importing Pre-Requisite Libraries/Packages
import os
import ee
import geemap
import geopandas as gpd
import pandas as pd
import glob
from bug_detect_US import zonal_stat_calc
# ee.Authenticate() # only one time authorization required
ee.Initialize()


def read_datasets(start_date, end_date, mask_year, CloudCoverMax, shape_file=None):
    # Applicable only for US area
    lc_s2 = ee.ImageCollection("USFS/GTAC/LCMS/v2021-7") .filter(ee.Filter.eq('year', mask_year)).first()
    naip_2020 = ee.ImageCollection("USDA/NAIP/DOQQ").filter(ee.Filter.date(start_date, end_date))
    # Ingest the shapefiles - US state
    shp = geemap.geopandas_to_ee(shape_file) # State

    # Sentinel-2 surface reflectance image
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CloudCoverMax)) \
        .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', CloudCoverMax)) \
        .filterDate(start_date, end_date)\
        .filterBounds(shp)
    print('The Size of the Sentinel-2 image collection is :',s2_sr.size().getInfo())
    print('1. Reading S2 imagecollection, 10 m landuse land cover and shapefile - completed')
    return s2_sr, naip_2020, lc_s2

def data_preprocessing(s2_sr, lc_s2):
    from bug_detect_US import scale_band_new
    s2_sr = s2_sr.map(scale_band_new)
    # Change the band names
    inBands = ee.List(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']);
    outBands = ee.List(['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'rededge4', 'waterVapor', 'swir1', 'swir2']);
    s2_sr = s2_sr.select(inBands, outBands)
    # mask the non-forest class
    s2_sr_mean = s2_sr.mean()
    forest_mask = lc_s2.select('Land_Use').eq(3)
    s2_sr_mean = s2_sr_mean.updateMask(forest_mask)
    print('2.1 Data Pre-processing- Scaling S2 images, changing band names, Image composite generation and masking non-forest areas- Completed')
    return s2_sr_mean

def reproject_utm(shape_utm,epsg):
    shape_utm['geometry'] = shape_utm['geometry'].to_crs(epsg=epsg)  # Sierra Nevada falls under UTM zone 11S
    return shape_utm


def calculate_indices(s2_sr_mean):
    from bug_detect_US import dvi, veg_ndvi, evi, veg_avi, veg_gndvi, veg_savi, veg_ndmi, veg_msi, veg_gci, soil_bsi, ndwi, ndsi, vari, ccci, chlgreen, lci, \
        ndre2, ndre3, cvi, gdvi, ng, ngdvi, bndvi, cig, gi, pbi, ci, pgr, dswi, vmi, lwci, rdi, sr_swir
    dvi = dvi(s2_sr_mean)
    ndvi = veg_ndvi(s2_sr_mean)
    evi = evi(s2_sr_mean)
    avi = veg_avi(s2_sr_mean)
    gndvi = veg_gndvi(s2_sr_mean)
    savi = veg_savi(s2_sr_mean)
    ndmi = veg_ndmi(s2_sr_mean)
    msi = veg_msi(s2_sr_mean)
    gci = veg_gci(s2_sr_mean)
    bsi = soil_bsi(s2_sr_mean)
    ndwi = ndwi(s2_sr_mean)
    ndsi = ndsi(s2_sr_mean)
    vari = vari(s2_sr_mean)

    # Pigmentation indices
    ccci = ccci(s2_sr_mean)
    chlgreen = chlgreen(s2_sr_mean)
    lci = lci(s2_sr_mean)
    ndre2 = ndre2(s2_sr_mean)
    ndre3 = ndre3(s2_sr_mean)
    cvi = cvi(s2_sr_mean)
    gdvi = gdvi(s2_sr_mean)
    ng = ng(s2_sr_mean)
    ngdvi = ngdvi(s2_sr_mean)
    bndvi = bndvi(s2_sr_mean)
    cig = cig(s2_sr_mean)
    gi = gi(s2_sr_mean)
    pbi = pbi(s2_sr_mean)
    ci = ci(s2_sr_mean)
    pgr = pgr(s2_sr_mean)

    # Water Indices
    dswi = dswi(s2_sr_mean)
    vmi = vmi(s2_sr_mean)
    lwci = lwci(s2_sr_mean)
    rdi = rdi(s2_sr_mean)
    sr_swir = sr_swir(s2_sr_mean)

    indices_collection = [dvi, ndvi, evi, avi, gndvi, savi, ndmi, msi, gci, bsi, ndwi, ndsi, vari, ccci, chlgreen, lci,
                          ndre2, ndre3, cvi, gdvi, ng, ngdvi, bndvi, cig, gi, pbi, ci, pgr, dswi, vmi, lwci, rdi, sr_swir]
    indices_list = ['DVI.csv', 'NDVI.csv', 'EVI.csv', 'AVI.csv', 'GNDVI.csv', 'SAVI.csv',
                    'NDMI.csv', 'MSI.csv', 'GCI.csv', 'BSI.csv', 'NDWI.csv', 'NDSI.csv', 'VARI.csv',
                    'CCCI.csv', 'CHLGREEN.csv', 'LCI.csv', 'NDRE2.csv', 'NDRE3.csv', 'CVI.csv',
                    'GDVI.csv', 'NG.csv', 'NGDVI.csv', 'BNDVI.csv', 'CIG.csv', 'GI.csv', 'PBI.csv',
                    'CI.csv', 'PGR.csv', 'DSWI.csv', 'VMI.csv', 'LWCI.csv', 'RDI.csv', 'SR_SWIR.csv']
    print('2.2 Data Pre-processing- indices generation- Completed')
    return indices_collection, indices_list

def extract_indices(buffers, indices_list, indices_collection, buff_list, out_dir, shape_file):
    from bug_detect_US import zonal_stat_calc
    zonal_stat_calc(buffers, out_dir, indices_list, indices_collection, shape_file, buff_list)

    # zonal_stat_calc(buffers, buff_list, out_dir_healthy, shape_file=healthy_utm, indices_list=indices_list,
    #                 indices_collection=indices_collection)
    print('2.3 extraction of values for labelled data - completed')

def combine_csv(out_dir, label):
    path = os.path.join(out_dir,'Buff_10')
    files = glob.glob(path+"/*.csv")
    comb_csv = pd.concat([pd.read_csv(f).filter(like='mean') for f in files], axis=1)
    data_list = []
    for i in range(0,len(files)):
        x = files[i].split("/")
        y= (x[-1])
        z = y[:-4]
        data_list.append(z)
    comb_csv.columns = data_list
    comb_csv['label']= label
    comb_csv.head()
    print('2.4. All individual predictor variables are concatenated to form a DataFrame')
    return comb_csv


if __name__ == '__main__':
    # 1. Priori parameters
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    mask_year = 2020
    CloudCoverMax = 10
    buffers = [10]
    buff_list = ['Buff_10']
    # 1.1 Shapefiles
    us_states = gpd.read_file('data/US_shapefiles/us-states.shp')
    shape_file = us_states[us_states['name'] == 'California'] # Select appropriate state for analysis
    bug_shape = gpd.read_file('data/shapefiles/bug_infested_2020.shp')
    healthy_shape = gpd.read_file('data/shapefiles/healthy_2020.shp')
    # Reprojecting the shapefiles to create buffers
    bug_utm = reproject_utm(shape_utm=bug_shape, epsg=32711)
    healthy_utm = reproject_utm(shape_utm=healthy_shape, epsg=32711)
    print('all priori data read completed')

    # 2. Data processing and analysis
    # 2.1 Data Pre-processing- Scaling S2 images, changing band names, Image composite generation and masking non-forest areas
    s2_sr, naip_2020, lc_s2 = read_datasets(start_date, end_date, mask_year, CloudCoverMax, shape_file=shape_file)
    s2_sr_mean = data_preprocessing(s2_sr, lc_s2)
    # 2.2 Data Pre-processing- indices generation
    indices_collection, indices_list = calculate_indices(s2_sr_mean)
    # 2.3 Data Pre-processing- zonal statistics for bug infested and healthy vegetation (Issue with gee authentication)
    out_dir_b = 'data/data_extract/bug_infested/'
    out_dir_h = 'data/data_extract/healthy/'
    ## NOTE: RUN THIS CODE ONLY ONCE- DOWNLOAD DATA AND COMMENT THE FOLLOWING LINES ######################
    # from bug_detect_US import zonal_stat_calc
    zonal_stat_calc(buffers=buffers, out_dir=out_dir_b, shape_file=bug_utm, indices_list=indices_list,
                    indices_collection=indices_collection, buff_list=buff_list)

    zonal_stat_calc(buffers=buffers, out_dir=out_dir_h, shape_file=healthy_utm, indices_list=indices_list,
                    indices_collection=indices_collection, buff_list=buff_list)
    ####################################################################################################
    print('2.3 Zonal Statistics for 10 m buffer calculated')
    # 2.4 - Combine all the CSV files into one and label the dataframe
    comb_inf = combine_csv(out_dir=out_dir_b, label='Bug_infested')
    comb_healthy = combine_csv(out_dir=out_dir_h, label='Healthy')
    # 2.5 Save the concatenated data frame as a csv file- for ML model development
    comb_inf.to_csv('data/data_extract/indices_mean_buginf_2020.csv')
    comb_healthy.to_csv('data/data_extract/indices_mean_healthy_2020.csv')










