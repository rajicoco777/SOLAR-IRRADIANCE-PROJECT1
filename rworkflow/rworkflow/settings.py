from datetime import *
import os
import numpy as np
#from oct2py import Oct2Py, octave

#os.environ['OCTAVE_EXECUTABLE']='C:\\Octave\\Octave-4.2.1\\bin\\octave-cli.exe'
#octave.addpath('C:\\Octave\\Octave-4.2.1\\share\\octave\\4.2.1\\m')

path = f'{os.sep}'.join(os.path.dirname(__file__).split(os.sep)[:-1])
partialOutputPath = f'{path}{os.sep}partial_output{os.sep}'
outputPath = f'{path}{os.sep}output{os.sep}'
#pathReplace = path+'Substitutos\\'
figuresPath = f'{path}{os.sep}figures{os.sep}'

# #212 days
# initialDate = '01/02/2011 00:00:00'
# finalDate = '31/08/2011 23:45:00'

# #106 days
# initialDate = '31/01/2011 00:00:00'
# finalDate = '16/05/2011 23:45:00'

# #106 days
# initialDate = '18/05/2011 00:00:00'
# finalDate = '31/08/2011 23:45:00'

# 413 days (TSI and SSI)
#initialDate = '14/09/2011 00:00:00'
#finalDate = '30/10/2012 23:45:00'

# # 29 days
# initialDate = '20/11/2012 00:00:00'
# finalDate = '18/12/2012 23:45:00'

#301 days (TSI and SSI)
# initialDate = '05/03/2014 00:00:00'
# finalDate = '31/12/2014 23:45:00'

#Período de Treinamento para SSI
# 130 days
# initialDate = '15/12/2012 00:00:00'
# finalDate = '23/04/2013 23:45:00'

#Período de Treinamento para SSI
#82 days
# initialDate = '25/04/2013 00:00:00'
# finalDate = '15/07/2013 23:45:00'

# #Períodos para Testes durante a escrita do documento
initialDate = '01/10/2014 00:00:00'
finalDate = '31/10/2014 23:45:00'

# initialDate = '22/10/2014 00:00:00'
# finalDate = '22/10/2014 01:00:00'

dt = 1 #data resolution in days
rLabel = str(int(24*dt))+'h.csv'        

initialDate = datetime.strptime(initialDate,'%d/%m/%Y %H:%M:%S')
finalDate = datetime.strptime(finalDate,'%d/%m/%Y %H:%M:%S')

tHours = initialDate.hour/24
tMinutes = initialDate.minute/(24*60)
tSeconds = initialDate.second/(24*60*60)
dI = initialDate.toordinal()+tHours+tMinutes+tSeconds

tHours = finalDate.hour/24
tMinutes = finalDate.minute/(24*60)
tSeconds = finalDate.second/(24*60*60)
dF = finalDate.toordinal()+tHours+tMinutes+tSeconds

outputLabel = str(int(dF-dI))+'dias'

resolution = 60*(24*dt) #in minutes

url = "http://jsoc.stanford.edu/data/hmi/images"

#if dt<1:
tsiFile = 'sorce_tsi_L3_c06h_latest.txt'
#elif dt==1:
#tsiFile = 'sorce_tsi_L3_c24h_latest.txt'
    
# SSILines = np.array([30.5, 48.5, 121.5])
SSILines = np.array([549.91,698.85,798.83])
#SSILines = np.array([121.6])

fnamesEve = []
fnamesSorce = [path + 'sorce_sim_ssi_l3_549.41.csv', 
               path + 'sorce_sim_ssi_l3_698.85.csv',
               path + 'sorce_sim_ssi_l3_798.83.csv']
# fnamesEve = [path + 'sdo_eve_ssi_48.5.csv']

continuumSufix = '_Ic_flat_1k.jpg'
magSufix = '_M_1k.jpg'

# magPath = path + 'mag 01.Jan a 31.Ago 2011\\'
# continuumPath = path + 'continuum 01.Jan a 31.Ago 2011\\'

#magPath = path + 'mag 14.Set.2011 a 30.Out.2012\\'
#continuumPath = path + 'continuum 14.Set.2011 a 30.Out.2012\\'

# magPath = path + 'mag 2014\\'
# continuumPath = path + 'continuum 2014\\'

# magPath = path + 'mag 20.Nov.2012 a 15.Jul.2013\\'
# continuumPath = path + 'continuum 20.Nov.2012 a 15.Jul.2013\\'

# magPath = './continuum 20.Nov.2012 a 15.Jul.2013//'
# continuumPath = './continuum 20.Nov.2012 a 15.Jul.2013//'

magPath = f'{path}{os.sep}data{os.sep}mag{os.sep}'
continuumPath = f'{path}{os.sep}data{os.sep}continuum{os.sep}'

# sPathMag = pathMag + 'Substitutos\\'
# sPathContinuum = pathContinuum + 'Substitutos\\'  
