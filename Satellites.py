from keras.models import Sequential, Model
from keras.layers import Convolution2D, Activation, Dropout, Deconvolution2D, MaxPooling2D, UpSampling2D

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch

import matplotlib.pyplot as plt




model = Sequential()

model.add(Convolution2D(16, 5, 5, input_shape = (3, None, None)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))

#model.add(Deconvolution2D(32, 5, 5, output_shape = (16, None, None), subsample=(1,1)))
#model.add(Activation('relu'))
#model.add(UpSampling2D((2, 2)))
#model.add(Deconvolution2D(16, 5, 5))
#model.add(Activation('relu'))

#model.add((Convolution2D(10, 3, 3, border_mode = 'same')))
#model.add(Activation('sigmoid'))

inDir='../db'

def get_image_names(imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': '{}/three_band/{}.tif'.format(inDir, imageId),
         'A': '{}/sixteen_band/{}_A.tif'.format(inDir, imageId),
         'M': '{}/sixteen_band/{}_M.tif'.format(inDir, imageId),
         'P': '{}/sixteen_band/{}_P.tif'.format(inDir, imageId),
         }
    return d

def get_images(imageId, img_key = None):
    '''
    Load images correspoding to imageId
	
    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    img_key : {None, '3', 'A', 'M', 'P'}, optional
        Specify this to load single image
        None loads all images and returns in a dict
        '3' loads image from three_band/
        'A' loads '_A' image from sixteen_band/
        'M' loads '_M' image from sixteen_band/
        'P' loads '_P' image from sixteen_band/

    Returns
    -------
    images : dict
        A dict of image data from TIFF files as numpy array
    '''
    img_names = get_image_names(imageId)
    images = dict()
    if img_key is None:
        for k in img_names.keys():
            images[k] = tiff.imread(img_names[k])
    else:
        images[img_key] = tiff.imread(img_names[img_key])
    return images

x = get_images('6110_2_2')

def build_objective(im_id, df):
	x = get_images(im_id)['3']
	im_obj = np.zeros((10, x.shape[1], x.shape[2]))

	for c in range(10):
		print(c)
		polygon = wkt_loads(df[df['ImageId'] == im_id].MultipolygonWKT.values[c])
		if polygon.bounds != ():
			print(polygon.bounds)
			for i in range(int(x.shape[1]*polygon.bounds[0]), int(x.shape[1] * polygon.bounds[2])):
				for j in range(int(x.shape[2] * polygon.bounds[1]), int(x.shape[2] * polygon.bounds[3])):
					p = Point(i/x.shape[1], j/x.shape[2])
					if polygon.contains(p):
						im_obj[c,i,-j] = 1.0

	return im_obj


