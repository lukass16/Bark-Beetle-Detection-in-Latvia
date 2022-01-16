# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:23:23 2021

This is a summary of code and used methods for the scientific study of forest damage-bark beetle 
detection using remote sensing in Latvia - this is not a functional script nor it is aimed to be one - 
for functioning code see the separate ipynb/html and python files listed.

@author: Lukass Roberts Kellijs
"""


from osgeo import gdal
import rasterio #https://github.com/rasterio/rasterio
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import cv2


###-------Necessary structure, method and function definitions-------###

class Region:
    def __init__(self, label, xmin, ymin, xmax, ymax):
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    def info(self):
        info_str = self.label + " " + str(self.xmin) + " " + str(self.ymin) + " " + str(self.xmax) + " " + str(self.ymax)
        return info_str
    def set_roi(self, img):
        roi = img[self.ymin : self.ymax, self.xmin : self.xmax, :]
        self.roi = np.array(roi, dtype="int32")
        return roi
    def show_roi(self, it):
        plt.figure(it)
        plt.imshow(np.dstack((self.roi[:,:,0], self.roi[:,:,1], self.roi[:,:,2])))
        plt.title(self.label)
        
def getAllPixels(array):
    pixels = list()
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            pixels.append(array[y][x][:])      
    return pixels

def removeMaskedPixels(unmasked_pixels):
    pixels = list()
    for pixel in unmasked_pixels:
        if pixel[0] != 0 and pixel[1] != 0 and pixel[2] != 0 and pixel[3] != 0:
            pixels.append(pixel)
    return pixels

def reorderBands(img):
    #Band order currently: b g n r
    red = img[:,:,3]
    green = img[:,:,1]
    blue = img[:,:,0]
    nir = img[:,:,2]
    img = np.dstack((red, green, blue, nir))
    return img

def getSmallDF(infected_pixels, healthy_pixels):
    #What we need
    i_avg_red = np.average(infected_pixels[:, 0])
    i_avg_green = np.average(infected_pixels[:, 1])
    i_avg_blue = np.average(infected_pixels[:, 2])
    i_avg_nir = np.average(infected_pixels[:, 3])
    h_avg_red = np.average(healthy_pixels[:, 0])
    h_avg_green = np.average(healthy_pixels[:, 1])
    h_avg_blue = np.average(healthy_pixels[:, 2])
    h_avg_nir = np.average(healthy_pixels[:, 3])

    i_std_red = np.std(infected_pixels[:, 0])
    i_std_green = np.std(infected_pixels[:, 1])
    i_std_blue = np.std(infected_pixels[:, 2])
    i_std_nir = np.std(infected_pixels[:, 3])
    h_std_red = np.std(healthy_pixels[:, 0])
    h_std_green = np.std(healthy_pixels[:, 1])
    h_std_blue = np.std(healthy_pixels[:, 2])
    h_std_nir = np.std(healthy_pixels[:, 3])

    small_data = [["infected", i_avg_red, i_avg_green, i_avg_blue, i_avg_nir, i_std_red, i_std_green, i_std_blue, i_std_nir], ["healthy", h_avg_red, h_avg_green, h_avg_blue, h_avg_nir, h_std_red, h_std_green, h_std_blue, h_std_nir]]
    small_df = pd.DataFrame(data=np.array(small_data), columns="Label Average-red Average-green Average-blue Average-nir Std-dev-red Std-dev-green Std-dev-blue Std-dev-nir".split())
    return small_df

def linearize(merged):
    height = merged.shape[0]
    width = merged.shape[1]
    return np.reshape(merged, (height*width, 4), order='F')

def delinearize(linearized, height, width):
    return np.reshape(linearized, (height, width), order='F')

def mapBeetle(merged, rfc):
    #receives as input a merged image of bands in form of numpy arrays in the following order: r g b n
    height = merged.shape[0]
    width = merged.shape[1]
    lin = linearize(merged)
    pred = rfc.predict(lin)
    delin = delinearize(pred, height, width)
    return delin

###----------------------------------------------------------------###

#loading RGB ortophoto image
rgb_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/VIS_clipped.tif"

vis = rasterio.open(rgb_path)
rgb = vis.read()
rgb = np.moveaxis(rgb, 0, 2) #tranfsorms into wanted shape


###-------Creating shadow masks-------###

red = vis.read(1)
green = vis.read(1)
blue = vis.read(1)

#tresholding shadow values based on blue layer
ret, mask67 = cv2.threshold(blue, 67, 255, cv2.THRESH_BINARY)
masked67 = rgb.copy()

#applying mask
masked67[mask67==0] = 0

#saving mask to tif file
filepath = "D:/Projekti/ZPD\QGIS/LGIA data/3334-42_3 roi/VIS_clipped_dis.tif"
ds = gdal.Open(filepath)
gt = ds.GetGeoTransform()
proj = ds.GetProjection()
print(gt, "\n", proj)

driver = gdal.GetDriverByName("GTiff")
driver.Register()

outds = driver.Create("mask67.tif", xsize = mask67.shape[1],
                      ysize = mask67.shape[0], bands = 1, 
                      eType = gdal.GDT_Int16)
outds.SetGeoTransform(gt)
outds.SetProjection(proj)

outband1 = outds.GetRasterBand(1)
outband1.WriteArray(mask67)
outband1.SetNoDataValue(np.nan)
outband1.FlushCache()
outband1 = None
outds = None

###-----------------------------------###



###-------Creating clearing masks-------###

#loading digital height grid model tif file (abbreviated as las)
las_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/lasgrid3.tif"

las_d = rasterio.open(las_path)
las = las_d.read(1)

#applying sequential median blurs
med1 = cv2.medianBlur(las,5)
med2 = cv2.medianBlur(med1,5)
med3 = cv2.medianBlur(med2,5)

#masking forrest clearings
ret, mask = cv2.threshold(med3, 135, 255, cv2.THRESH_BINARY)

#resizing mask to match size of otrophoto
resized_mask = cv2.resize(mask, (rgb.shape[0], rgb.shape[1]))

#saving mask
filepath = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/VIS_clipped.tif"
ds = gdal.Open(filepath)
gt = ds.GetGeoTransform()
proj = ds.GetProjection()
print(gt, "\n", proj)

driver = gdal.GetDriverByName("GTiff")
driver.Register()

outds = driver.Create("lasmask.tif", xsize = resized_mask.shape[1],
                      ysize = resized_mask.shape[0], bands = 1, 
                      eType = gdal.GDT_Int16)
outds.SetGeoTransform(gt)
outds.SetProjection(proj)

outband1 = outds.GetRasterBand(1)
outband1.WriteArray(resized_mask)
outband1.SetNoDataValue(np.nan)
outband1.FlushCache()
outband1 = None
outds = None

###-------------------------------------###


'''
NOTE: After this step the ortophoto files are combined with the masks into a merged masked image using QGis
     - this could be done with the help of python, however it is crucial to test and display the results of the masks and their combination
'''


###----------Opening merged tif file, extracting regions of interest---------###

#file path definition
merged_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/merged_clipped.tif"
xml_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/VIS_clipped.xml"

#opening ortophotos - merged and VIS
merged = rasterio.open(merged_path)
img = merged.read()
img = np.moveaxis(img, 0, 2) #tranfsorms into wanted shape
img = reorderBands(img)



#opening xml file containing labeled areas
xmltree = ET.parse(xml_path)
root = xmltree.getroot()

regions = list()
rois = list()
infected_list = list()
healthy_list = list()

#creating a list of labelled regions
for obj in root.findall("object"):
    name = obj.find("name")
    bndbox = obj.find("bndbox")
    regions.append(Region(name.text, int(bndbox[0].text), int(bndbox[1].text), int(bndbox[2].text), int(bndbox[3].text)))
    roi = regions[-1].set_roi(img)
    #regions[-1].show_roi(len(regions)-1) #--uncomment this line to see all regions of interest

###--------------------------------------------------------------------------###



###-------Get Small Data Frame For Primary Observation-------###

#for all labelled regions create a list for infected and healthy pixels
for region in regions:
    if region.label == "infected":
        pixels = getAllPixels(region.roi)
        pixels = removeMaskedPixels(pixels)
        for pixel in pixels:    
            infected_list.append([pixel[0], pixel[1], pixel[2], pixel[3]]) 
    elif region.label == "healthy":
        pixels = getAllPixels(region.roi)
        pixels = removeMaskedPixels(pixels)
        for pixel in pixels:    
            healthy_list.append([pixel[0], pixel[1], pixel[2], pixel[3]])

infected_pixels = np.array(infected_list)
healthy_pixels = np.array(healthy_list)

#create a small dataframe with average values indicative of dataset
small_df = getSmallDF(infected_pixels, healthy_pixels)
x = small_df.drop('Label', axis=1)
y = small_df['Label']

#save to excel sheet
small_df.to_excel('small_df_train.xlsx',sheet_name='small_df')

###---------------------------------------------------------###



###-------Get Large Data Frame-------###

data = list()
for region in regions:
    region_data = list()
    pixels = getAllPixels(region.roi)
    pixels = removeMaskedPixels(pixels)
    for pixel in pixels:
        region_data.append([region.label, pixel[0], pixel[1], pixel[2], pixel[3]])
    data += region_data

#creating large dataframe (detailed in the study)
large_df = pd.DataFrame(data=data, columns = "Label Red Green Blue Nir".split())

#displaying spectral differences for healthy and infected trees in the Nir and Red bands
plt.figure(figsize=(8,4)) 
sns.boxplot(x='Label', y="Nir",data=large_df,palette="rainbow")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x='Label', y="Red",data=large_df,palette="rainbow")
plt.show()

#preparing datasets for training
X = large_df.drop('Label',axis=1)
Y = large_df['Label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

large_df.to_excel('large_df_train.xlsx',sheet_name='large_df')

###---------------------------------###

#Creating random forest classifier
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, Y_train)

#making predictions - evaluating model
rfc_pred = rfc.predict(X_test)
print(classification_report(Y_test,rfc_pred))
print(confusion_matrix(Y_test,rfc_pred))

#Save model
save_name = "rfc_bark_beetle_model.sav"
pickle.dump(rfc, open(save_name, "wb"))

#show results of model
confusion_table = pd.DataFrame(data=confusion_matrix(Y_test,rfc_pred),columns="Vesels Nevesels".split(),index="Vesels Nevesles".split())
ax = sns.heatmap(confusion_table,cmap='coolwarm',annot=True)
ax.set_xlabel("Patiesās vērtības")
ax.set_ylabel("Prognozētās vērtības")



###-------Use model to map Bark Beetle attack on entire study area-------###

save_name = "mappedBeetleEntire_np.sav"
mappedBeetle = mapBeetle(rgb, rfc)
pickle.dump(mappedBeetle, open(save_name, "wb"))

#creating categorical image of infestation, infested values are marked flagged 1 
out = np.zeros(mappedBeetle.shape)
print(mappedBeetle[0])
out[mappedBeetle == "infected"] = 1

#adding mask
mask_df_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/full_mask.tif"
mask_df = rasterio.open(mask_df_path)
mask = mask_df.read(1)

out[mask == 0] = 0
to_blur = np.array(out, dtype="float32")

#applying median blur to filter out individual noise pixels
med = cv2.medianBlur(to_blur,3)
med = cv2.medianBlur(med,3)
med = cv2.medianBlur(med,3)

#applying threshold
ret, thresh = cv2.threshold(med, 0.9, 1, cv2.THRESH_BINARY)

#
red = rgb[:,:,0].copy()
green = rgb[:,:,1].copy()
blue = rgb[:,:,2].copy()

#set infected colors 255, 61, 36 
red[out == 1] = 255
green[out == 1] = 61
blue[out == 1] = 36

#attain result
result = np.dstack((red,green,blue)).astype(np.int32)

#show result
fig = plt.imshow(result)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig("result.png")

#save result as tif file
filepath = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/VIS_clipped.tif"
ds = gdal.Open(filepath)
gt = ds.GetGeoTransform()
proj = ds.GetProjection()
print(gt, "\n", proj)

driver = gdal.GetDriverByName("GTiff")
driver.Register()

outds = driver.Create("resultROIVISRed.tif", xsize = result.shape[1],
                      ysize = result.shape[0], bands = 3, 
                      eType = gdal.GDT_Int16)
outds.SetGeoTransform(gt)
outds.SetProjection(proj)

outband1 = outds.GetRasterBand(1)
outband2 = outds.GetRasterBand(2)
outband3 = outds.GetRasterBand(3)

outband1.WriteArray(red)
outband1.SetNoDataValue(np.nan)
outband1.FlushCache()
outband1 = None
outband2.WriteArray(green)
outband2.SetNoDataValue(np.nan)
outband2.FlushCache()
outband2 = None
outband3.WriteArray(blue)
outband3.SetNoDataValue(np.nan)
outband3.FlushCache()
outband3 = None
outds = None

###----------------------------------------------------------------------###