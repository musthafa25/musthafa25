# Importing Pre-Requisite Libraries/Packages
# indices_list
# mask function
import os
import ee
import geemap
import geopandas as gpd
import pandas as pd


def forest_mask(image, lc_s2):
    forest_mask = lc_s2.select('Land_Use').eq(3)  # In land use forest = 3
    return image.updateMask(forest_mask)


# Sentinel_indices
def dvi(image):
    dvi = image.expression(('nir-red'), {
        'nir': image.select('nir'),
        'red': image.select('red')}).rename('dvi')
    return dvi


def veg_ndvi(image):
    ndvi = image.expression('(nir-red)/(nir+red)', {
        'nir': image.select('nir'),
        'red': image.select('red')}).rename('ndvi')
    return ndvi


def evi(image):
    evi = image.expression('2.5*((nir-red)/(nir+6*red-7.5*blue+1))', {
        'nir': image.select('nir'),
        'red': image.select('red'),
        'blue': image.select('blue')}).rename('evi')
    return evi


def veg_gndvi(image):
    gndvi = image.expression('(nir-green)/(nir+green)', {
        'nir': image.select('nir'),
        'green': image.select('green')}).rename('gndvi')
    return gndvi


def veg_avi(image):  # ADVANCED VEGETATION INDEX
    avi = image.expression('(nir*(1-red)*(nir-red))**(1./3)', {
        'nir': image.select('nir'),
        'red': image.select('red')}).rename('avi')
    return avi


def veg_savi(image):
    savi = image.expression('((nir-red)/(nir+red+L))*(1+L)', {
        'nir': image.select('nir'),
        'red': image.select('red'),
        'L': 1.5}).rename('savi')
    return savi


def veg_ndmi(image):  # Normalized difference moisture index
    ndmi = image.expression('(nir-swir)/(nir+swir)', {
        'nir': image.select('nir'),
        'swir': image.select('swir1')}).rename('ndmi')
    return ndmi


def veg_msi(image):  # moisture stress index
    msi = image.expression('midIR/nir', {
        'nir': image.select('nir'),
        'midIR': image.select('swir1')}).rename('msi')
    return msi


def veg_gci(image):  # green cover index
    gci = image.expression('nir/green-1', {
        'nir': image.select('nir'),
        'green': image.select('green')}).rename('gci')
    return gci


# Soil index
def soil_bsi(image):  # bare soil index
    bsi = image.expression('((red+swir)-(nir+blue))/((red+swir)+(nir+blue))', {
        'red': image.select('red'),
        'nir': image.select('nir'),
        'swir': image.select('swir1'),
        'blue': image.select('blue')}).rename('bsi')
    return bsi


# Water index
def ndwi(image):
    ndwi = image.expression('(nir-swir)/(nir+swir)', {
        'nir': image.select('nir'),
        'swir': image.select('swir1')}).rename('nswi')
    return ndwi


# Snow Index
def ndsi(image):
    ndsi = image.expression('(nir-green)/(nir+green)', {
        'nir': image.select('nir'),
        'green': image.select('green')}).rename('ndsi')
    return ndsi


# Visual atmospheric resistance index
def vari(image):
    vari = image.expression('(green-red)/(green+red-blue)', {
        'red': image.select('red'),
        'green': image.select('green'),
        'blue': image.select('blue')}).rename('vari')
    return vari


# new sentinel-2 indices - chlorophyll pigments, greeness
def ccci(image):  # canopy chlorophyll content index
    ccci = image.expression('((nir-rededge1)/(nir+rededge1))/((nir-red)/(nir+red))', {
        'red': image.select('red'),
        'nir': image.select('nir'),
        'rededge1': image.select('rededge1')}).rename('ccci')
    return ccci


def chlgreen(image):  # Chlorophyll green
    chlgreen = image.expression('(rededge3/green)**(-1)', {
        'rededge3': image.select('rededge3'),
        'green': image.select('green')}).rename('chlgreen')
    return chlgreen


def lci(image):  # Leaf chlorophyll index
    lci = image.expression('(nir-rededge1)/(nir+red)', {
        'red': image.select('red'),
        'nir': image.select('nir'),
        'rededge1': image.select('rededge1')}).rename('lci')
    return lci


def ndre2(image):  # normalized difference red-edge2
    ndre2 = image.expression('(nir-rededge1)/(nir+rededge1)', {
        'nir': image.select('nir'),
        'rededge1': image.select('rededge1')}).rename('ndre2')
    return ndre2


def ndre3(image):  # normalized difference red-edge3
    ndre2 = image.expression('(nir-rededge2)/(nir+rededge2)', {
        'nir': image.select('nir'),
        'rededge2': image.select('rededge2')}).rename('ndre2')
    return ndre2


def cvi(image):  # chlorophyll vegetation index
    cvi = image.expression('nir*(red/green**2)', {
        'nir': image.select('nir'),
        'green': image.select('green'),
        'red': image.select('red')}).rename('cvi')
    return cvi


def gdvi(image):  # green difference vegetation index
    gdvi = image.expression('nir-green', {
        'nir': image.select('nir'),
        'green': image.select('green')}).rename('gdvi')
    return gdvi


def ng(image):  # normalize green
    ng = image.expression('green/(nir+red+green)', {
        'green': image.select('green'),
        'red': image.select('red'),
        'nir': image.select('nir')}).rename('ng')
    return ng


def ngdvi(image):  # normalized difference green/red normalized green red difference index
    ngdvi = image.expression('(green-red)/(green+red)', {
        'green': image.select('green'),
        'red': image.select('red')}).rename('ngdri')
    return ngdvi


def bndvi(image):  # normalized difference nir/blue bluenormalized differencevegetation index
    bndvi = image.expression('(nir-blue)/(nir+blue)', {
        'nir': image.select('nir'),
        'blue': image.select('blue')}).rename('bndvi')
    return bndvi


def cig(image):  # chlorophyll index green
    cig = image.expression('(nir/green)-1', {
        'nir': image.select('nir'),
        'green': image.select('green')}).rename('cig')
    return cig


def gi(image):  # green index
    gi = image.expression('green/red', {
        'green': image.select('green'),
        'red': image.select('red')}).rename('gi')
    return gi


def pbi(image):  # plant biochemical index
    pbi = image.expression('nir/green', {
        'green': image.select('green'),
        'nir': image.select('nir')}).rename('pbi')
    return pbi


def ci(image):  # coloration index
    ci = image.expression('(red-blue)/red', {
        'red': image.select('red'),
        'blue': image.select('blue')}).rename('ci')
    return ci


def pgr(image):  # plant pigment ratio
    pgr = image.expression('(green-blue)/(green+blue)', {
        'green': image.select('green'),
        'blue': image.select('blue')}).rename('pgr')
    return pgr


# water indices
def dswi(image):  # disease stress water index
    dswi = image.expression('(nir+green)/(swir1+red)', {
        'nir': image.select('nir'),
        'green': image.select('green'),
        'red': image.select('red'),
        'swir1': image.select('swir1')}).rename('dswi')
    return dswi


def vmi(image):  # vegetation moisture index
    vmi = image.expression('((nir+0.1)-(swir1+0.02))/((nir+0.1)+(swir1+0.02))', {
        'nir': image.select('nir'),
        'swir1': image.select('swir1')}).rename('vmi')
    return vmi


def lwci(image):  # leaf water content index
    lwci = image.expression('(log(1-(nir-swir1)))/(-log(1-(nir-swir1)))', {
        'nir': image.select('nir'),
        'swir1': image.select('swir1')}).rename('lwci')
    return lwci


def rdi(image):  # ratio drounght index
    rdi = image.expression('swir1/nir', {
        'swir1': image.select('swir1'),
        'nir': image.select('nir')}).rename('rdi')
    return rdi


def sr_swir(image):  # simple ratio swir
    sr_swir = image.expression('swir1/swir2', {
        'swir1': image.select('swir1'),
        'swir2': image.select('swir2')}).rename('sr_swir')
    return sr_swir


# Get Sentinel-2 Image collection over a study site
def get_s2_collection(start_date, end_date, CloudCoverMax, x, y):
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CloudCoverMax)) \
        .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', CloudCoverMax)) \
        .filterDate(start_date, end_date) \
        .filterBounds(ee.Geometry.Point(x, y))
    return s2_sr

def get_s2_collection_all(start_date, end_date, CloudCoverMax,shp):
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CloudCoverMax)) \
        .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', CloudCoverMax)) \
        .filterDate(start_date, end_date)\
        .filterBounds(shp)
    return s2_sr

# Preprocess the Sentinel-2 data in the collection (scale the bands and change the names of bands into EM nomenclature)
# Scaling Sentinel-2A data
def scale_bands(img):
    prop = img.toDictionary()
    # t = (img.select(['QA60','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'])
    #      .divide(10000))
    t = (img.select(['QA60', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
         .divide(10000))
    t = (t.addBands(img.select(['QA60'])).set(prop)
         .copyProperties(img, ['system:time_start', 'system:footprint']))
    return ee.Image(t)


def scale_band_new(img):
    reflBands = (img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']).divide(10000))
    return reflBands



# %%writefile calculate_indices.py
def calculate_indices(data):
    # Indices calculation
    dvi_collection = data.map(dvi)
    ndvi_collection = data.map(veg_ndvi)
    evi_collection = data.map(evi)
    gndvi_collection = data.map(veg_gndvi)
    avi_collection = data.map(veg_avi)
    savi_collection = data.map(veg_savi)
    ndmi_collection = data.map(veg_ndmi)
    msi_collection = data.map(veg_msi)
    gci_collection = data.map(veg_gci)
    bsi_collection = data.map(soil_bsi)
    ndwi_collection = data.map(ndwi)
    ndsi_collection = data.map(ndsi)
    vari_collection = data.map(vari)

    # pigmentation indices
    ccci_collection = data.map(ccci)
    chlgreen_collection = data.map(chlgreen)
    lci_collection = data.map(lci)
    ndre2_collection = data.map(ndre2)
    ndre3_collection = data.map(ndre3)
    cvi_collection = data.map(cvi)
    gdvi_collection = data.map(gdvi)
    ng_collection = data.map(ng)
    ngdvi_collection = data.map(ngdvi)
    bndvi_collection = data.map(bndvi)
    cig_collection = data.map(cig)
    gi_collection = data.map(gi)
    pbi_collection = data.map(pbi)
    ci_collection = data.map(ci)
    pgr_collection = data.map(pgr)

    # water indices
    dswi_collection = data.map(dswi)
    vmi_collection = data.map(vmi)
    lwci_collection = data.map(lwci)
    rdi_collection = data.map(rdi)
    sr_swir_collection = data.map(sr_swir)
    return dvi_collection, ndvi_collection, evi_collection, gndvi_collection, avi_collection, savi_collection, \
           ndmi_collection, msi_collection, gci_collection, bsi_collection, ndwi_collection, ndsi_collection, vari_collection, \
           ccci_collection, chlgreen_collection, lci_collection, ndre2_collection, \
           ndre3_collection, cvi_collection, gdvi_collection, ng_collection, \
           ngdvi_collection, bndvi_collection, cig_collection, gi_collection, \
           pbi_collection, ci_collection, pgr_collection, dswi_collection, \
           vmi_collection, lwci_collection, rdi_collection, sr_swir_collection


# Calculate zonal statistics
def zonal_stat_calc(buffers, out_dir, shape_file, indices_list, indices_collection, buff_list=None):
    for i in range(0, len(buffers)):
        out_dir1 = os.path.expanduser(os.path.join(out_dir, buff_list[i]))
        temp_buff = shape_file.buffer(buffers[i])
        temp_buff = gpd.GeoDataFrame(geometry=gpd.GeoSeries(temp_buff))
        temp_buff['geometry'] = temp_buff['geometry'].to_crs(
            epsg=4326)  # error is - it formed GeoSeries instead of GeoDataFrame (rectify it)
        temp_buff1 = geemap.geopandas_to_ee(temp_buff)
        for j in range(0, len(indices_list)):
            geemap.zonal_statistics(indices_collection[j], temp_buff1, os.path.join(out_dir1, indices_list[j]),
                                    statistics_type='MEAN', scale=buffers[i], maxPixels=1e13)

# Create column list for a set of analysis
def create_col_list(inDir,year):
    tem = os.listdir(os.path.join(inDir,'Buff_10'))
    dt = os.path.join('Buff_10',tem[0])
    dt_avi = pd.read_csv(os.path.join(inDir, dt)).filter(like= year)
    cols = dt_avi.columns.to_list()
    temp1 = [x for x in cols]   # Slice the column list containing only the imagery name
    col_list = []
    N = len(temp1)
    for i in range(0,N):                            # 5. Generate column list as dates for plotting function
        temp = temp1[i]
        temp2 = temp[0:len(temp)-36]
        col_list.append(temp2)
    return col_list


# Data REduction - through time-series - mean, std, min and max estimated
def data_reduction(timeseries, buffers, buff_list, inDir, year, label, col_list):
    for i in range(0, len(buffers)):
        path = os.path.expanduser(os.path.join(inDir, buff_list[i]))
        files = [x for x in os.listdir(path) if x.endswith('.csv')]
        for kk in range(0, len(files)):
            temp_kk = pd.read_csv(os.path.join(path, files[kk]))
            temp_kk = temp_kk.filter(like=year)
            test = temp_kk.iloc[:, 0:timeseries]
            test.columns = col_list
            tt = test.transpose()
            tt = tt.describe()
            df_ind = pd.DataFrame([tt.iloc[1, :], tt.iloc[2, :], tt.iloc[3, :], tt.iloc[7, :]])
            df_ind = df_ind.transpose()

            # Name the data description with respect to the indices
            t1 = files[kk][:-4] + '_mean'
            t2 = files[kk][:-4] + '_std'
            t3 = files[kk][:-4] + '_min'
            t4 = files[kk][:-4] + '_max'
            pp = [t1, t2, t3, t4]
            df_ind.columns = pp
            # Concatenate all the indices into a single csv file for each buffer
            if kk == 0:
                df = df_ind
            else:
                df = pd.concat([df, df_ind], axis=1)
        df['label'] = label
        file_name = 'stat' + buff_list[i] + '.csv'
        return df.to_csv(os.path.join(inDir, file_name))