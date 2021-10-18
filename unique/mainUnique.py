# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:27:13 2021

@author: Ami
"""

from libraryUnique import *
from allFunctionsUnique import *
from settingsUnique import *

dates = []

pasta = './mag'

for diretorio, subpastas, arquivos in os.walk(pasta):
    images = [arq for arq in arquivos if arq.lower().endswith(".jpg")]
    for a in images:
        data = a[:15].replace('_',' ')
        initialDate = datetime.strptime(data,"%Y%m%d %H%M%S")
        dates.append(initialDate)

for initialDate in dates:
    
    tHours = initialDate.hour/24
    tMinutes = initialDate.minute/(24*60)
    tSeconds = initialDate.second/(24*60*60)
    dI = initialDate.toordinal()+tHours+tMinutes+tSeconds

    geraAreas(initialDate)
    model_mdi_02_03(1, dI)
    output = TSI6HPrediction()