from astropy.io import fits

import cmath as cm

import cv2

import csv

from datetime import *

import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Dropout
from keras.callbacks import Callback

import julian

import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as clrs
from matplotlib import dates

import numpy as np
from numpy import matlib
from numpy import interp
import numpy.ma as ma

import os
import os.path
os.environ['OCTAVE_EXECUTABLE']='C:\\Octave\\Octave-4.2.1\\bin\\octave-cli.exe'
from oct2py import Oct2Py, octave
octave.addpath('C:\\Octave\\Octave-4.2.1\\share\\octave\\4.2.1\\m')

from pandas import read_csv

import pickle

from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.ndimage import correlate
from scipy import stats

from skimage.measure import label, regionprops
from scipy.interpolate import griddata
from skimage.segmentation import clear_border

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from urllib import request