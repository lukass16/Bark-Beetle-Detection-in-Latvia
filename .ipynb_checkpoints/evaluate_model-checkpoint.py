# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:23:23 2021

@author: Lukass
"""

#Useful tutorial: https://youtu.be/r6dyk68gymk

import rasterio #https://github.com/rasterio/rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import pickle
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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
        plt.imshow(np.dstack((self.roi[:,:,3], self.roi[:,:,1], self.roi[:,:,0])))
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

def mapBeetleByRow(merged, rfc):
    #receives as input a merged image of bands in form of numpy arrays in the following order: r g b n
    height = merged.shape[0]
    pred = list()
    for y in range(height):
        print("Progress: row " + str(y) + " - " + str(y/height, 2) + "%")
        row = rfc.predict(merged[y,:,:])
        pred.append(row)
    return pred

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

merged_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/merged_clipped.tif"
xml_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/VIS_clipped.xml"
merged = rasterio.open(merged_path)
img = merged.read()
img = np.moveaxis(img, 0, 2) #tranfsorms into wanted shape
img = reorderBands(img)

xmltree = ET.parse(xml_path)
root = xmltree.getroot()

regions = list()
rois = list()
infected_list = list()
healthy_list = list()

it = 0

#create a list of labelled regions
for obj in root.findall("object"):
    name = obj.find("name")
    bndbox = obj.find("bndbox")
    regions.append(Region(name.text, int(bndbox[0].text), int(bndbox[1].text), int(bndbox[2].text), int(bndbox[3].text)))
    roi = regions[-1].set_roi(img)
    it = it + 1
    #regions[-1].show_roi(it)
    #rois.append(roi)
    #print(roi.shape)



###-------Get Small Data Frame-------###

#for all labelled regions 
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

small_df = getSmallDF(infected_pixels, healthy_pixels)
x = small_df.drop('Label', axis=1)
y = small_df['Label']

###---------------------------------###


###-------Get Large Data Frame-------###
data = list()
for region in regions:
    region_data = list()
    pixels = getAllPixels(region.roi)
    pixels = removeMaskedPixels(pixels)
    for pixel in pixels:
        region_data.append([region.label, pixel[0], pixel[1], pixel[2], pixel[3]])
    data += region_data
    
large_df = pd.DataFrame(data=data, columns = "Label Red Green Blue Nir".split())
plt.figure(figsize=(8,4)) # this creates a figure 8 inch wide, 4 inch high
sns.boxplot(x='Label', y="Nir",data=large_df,palette="rainbow")
plt.show()

X = large_df.drop('Label',axis=1)
Y = large_df['Label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

###---------------------------------###

#Load Random Forest model
rfc = pickle.load(open("rfc_model.sav", 'rb'))

rfc_pred = rfc.predict(X_test)
print(classification_report(Y_test,rfc_pred))
print(confusion_matrix(Y_test,rfc_pred))

confusion_table = pd.DataFrame(data=confusion_matrix(Y_test,rfc_pred),columns="True False".split(),index="True False".split())
sns.heatmap(confusion_table,cmap='coolwarm',annot=True)


###-------Map Bark Beetle Predictions on Map-------###

#save_name = "mappedBeetleByRow_np.sav"
#mappedBeetleByRow = mapBeetleRowByRow(img, rfc)
#pickle.dump(mappedBeetleByRow, open(save_name, "wb"))
#mappedBeetleByRow = pickle.load(open(save_name, 'rb'))


save_name = "mappedBeetleEntire_np.sav"
#mappedBeetle = mapBeetle(img, rfc)
#pickle.dump(mappedBeetle, open(save_name, "wb"))
mappedBeetle = pickle.load(open(save_name, 'rb'))

#creating categorical image of infestation, infested values are marked flagged 1 
out = np.zeros(mappedBeetle.shape)
print(mappedBeetle[0])
out[mappedBeetle == "infected"] = 1

#adding mask
mask_df_path = "D:/Projekti/ZPD/QGIS/LGIA data/3334-42_3 roi/full_mask.tif"
mask_df = rasterio.open(mask_df_path)
mask = mask_df.read(1)

out[mask == 0] = 0
plt.figure(3)
plt.imshow(out)

save_name = "out_categorical.sav"
pickle.dump(out, open(save_name, "wb"))