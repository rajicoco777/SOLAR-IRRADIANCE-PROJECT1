# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:26:26 2021

@author: Ami
"""
from libraryUnique import *

path = f'{os.sep}'.join(os.path.dirname(__file__).split(os.sep)[:-1])
path = path+f'{os.sep}unique{os.sep}'
partialOutputPath = path+f'partial_output_files{os.sep}'
outputPath = path+f'output{os.sep}'

# finalDate = '22/10/2014 01:00:00'

dt = 1/4 #data resolution in days

# tHours = finalDate.hour/24
# tMinutes = finalDate.minute/(24*60)
# tSeconds = finalDate.second/(24*60*60)
# dF = finalDate.toordinal()+tHours+tMinutes+tSeconds

resolution = 60*(24*dt) #in minutes

continuumSufix = '_Ic_flat_1k.jpg'
magSufix = '_M_1k.jpg'

magPath = path+f'mag{os.sep}'
continuumPath = path+f'continuum{os.sep}'
