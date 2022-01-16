# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:23:23 2021

This is a script for the scientific study of forest damage-bark beetle detection using remote sensing in Latvia
This script was used to obtain necessary dataframes for the training and evaluation of a Random Forest model as well as for the model's creation

@author: Lukass
"""

#Useful tutorial: https://youtu.be/r6dyk68gymk

import rasterio #https://github.com/rasterio/rasterio
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
    #it = it + 1
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

small_df.to_excel('small_df_train.xlsx',sheet_name='small_df')

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

plt.figure(figsize=(8,4))
sns.boxplot(x='Label', y="Red",data=large_df,palette="rainbow")
plt.show()

X = large_df.drop('Label',axis=1)
Y = large_df['Label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

#large_df.to_csv('large_df_train.csv', index = False)

###---------------------------------###


#Random forest
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, Y_train)

rfc_pred = rfc.predict(X_test)
print(classification_report(Y_test,rfc_pred))
print(confusion_matrix(Y_test,rfc_pred))

#Save model
save_name = "rfc_model_dis.sav"
pickle.dump(rfc, open(save_name, "wb"))

confusion_table = pd.DataFrame(data=confusion_matrix(Y_test,rfc_pred),columns="Vesels Nevesels".split(),index="Vesels Nevesles".split())
ax = sns.heatmap(confusion_table,cmap='coolwarm',annot=True)
#ax.set_title("Apjukuma matrica")
ax.set_xlabel("Patiesās vērtības")
ax.set_ylabel("Prognozētās vērtības")
