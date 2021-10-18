# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:51:07 2021

@author: Ami
"""

from libraryUnique import *
from settingsUnique import *

def calc_mu_hmi(thresh):
        

    connectivity=8
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 100000 
    bw2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            bw2[output == i + 1] = 1
    
    labels = label(bw2)
    
    props = regionprops(labels)

    #Até aqui, todos os valores batem com o Matlab!!
    center_x = props[0]['Centroid'][0]
    center_y = props[0]['Centroid'][1]
    
    EquivDiameter = props[0]['EquivDiameter']
    
    mradius = EquivDiameter/2 
    
    #jx, jy = np.meshgrid(range(1,output.shape[0]+1), range(1,output.shape[1]+1)) #original
    jx, jy = np.meshgrid(range(output.shape[0]), range(output.shape[1]))
    
    jr = np.sqrt(np.power(jx-center_x,2) + np.power(jy-center_y,2))
    
    
    a = 1-np.power(jr/mradius,2)
    a = a.astype('complex')
    
    mu = np.real(np.sqrt(a))
    
    #mu = octave.real(octave.sqrt(a))
    
    ii0 = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    ii1 = np.arange(1,12) #original
    
    ii1 = ii1.astype(int)
    
    mu_rings = octave.interp1(ii0, ii1, mu, 'nearest')
    
    #mu_rings = interp1d(ii0, ii1, kind = 'nearest', fill_value = 'extrapolate')(mu)
 
    # As 10 linhas seguintes são a opção no caso de não utilizar o interpolador interp1.m do octave ou interp1d do Python.
    # mu_rings = np.empty_like(mu)
    # mu_rings[:] = np.nan
    
    # limites = [1, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.075]

    # for i in range(len(limites)-1):
    #     mu_rings[np.where((mu>limites[i+1]) & (mu<=limites[i]))] = ii1[i]
    
    # mu_rings[np.where((mu>=0) & (mu<limites[10]))] = ii1[10]
    
    
       
    return mu, mu_rings
    

def check_areas():
    
    #partialOutputPath= path + 'partial_output_files\\'

    time = np.loadtxt(partialOutputPath+'time.csv')
    area_c = np.loadtxt(partialOutputPath+'area_c.csv')
    alpha_mu_spot = np.loadtxt(partialOutputPath+'alpha_mu_spot.csv')
    
    alpha_mu_spot = np.reshape(alpha_mu_spot,(np.size(time),6,11))
        
    print('\nFiles saved in '+partialOutputPath+':\n')
    np.savetxt(partialOutputPath+'check_areas_time_PY.csv', [time])
    print('check_areas_time_PY.csv\n')
    
    np.savetxt(partialOutputPath+'check_areas_area_c_PY.csv', area_c)
    print('check_areas_area_c_PY.csv\n')
    
    with open(partialOutputPath+'check_areas_alpha_mu_spot_PY.csv', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(alpha_mu_spot.shape))
    
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for dataSlice in alpha_mu_spot:
    
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
    
    print('check_areas_alpha_mu_spot_PY.csv\n')

    return time, area_c, alpha_mu_spot


def continuumMasks(continuumImagePath,bw_mask):
    
    figuresPath = 'C:\\Users\Ami\Dropbox\Tese\Monografia\Figuras\\output\\'
    
    currentImage = cv2.imread(continuumImagePath)

    grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
    
    mu_rings = np.zeros_like(grayImage)
    mu_rings[np.where(grayImage>10)]=1
    
    ii = np.where(mu_rings>0)
    
    xMedian = np.nanmedian(grayImage[ii])
    xStd = np.nanstd(grayImage[ii])
    
    h = np.ones((5,5))/25
    I2 = convolve(grayImage,h)
    
    x2 = (I2 - xMedian)/xStd
    
    th1 = -4
    bw1 = np.zeros_like(x2)
    bw1[np.where(x2<=th1)] = 1
    
    bw1 = clear_border(bw1).astype(int)
    #bw1 = bw.astype(int)

    th2 = -15
    bw2 = np.zeros_like(x2)
    bw2[np.where(x2<=th2)] = 1

    bw2 = clear_border(bw2).astype(int) 
        
    bw_mask[np.where(bw1)] = 6
    bw_mask[np.where(bw2)] = 7
    
    fName = partialOutputPath+'bwMaskContinuum.csv'
    np.savetxt(fName, bw_mask)

    return bw_mask

def geraAreas(initialDate):

    currentDate = initialDate
    
    k=0
    
    aux_area = np.empty(6)
    aux_area.fill(np.nan)
    
    area_c = []
    
    aux_alpha_mu_spot = np.empty([6,11])
    aux_alpha_mu_spot.fill(np.nan)
    
    alpha_mu_spot = []
    
    time = []
    
    year = currentDate.strftime('%Y')
    month = currentDate.strftime('%m')
    day = currentDate.strftime('%d')
    hours = currentDate.strftime('%H')
    minutes = currentDate.strftime('%M')
    seconds = currentDate.strftime('%S')

    print('\nProcessing day '+str(currentDate))
    
    imageName = year + month + day + '_' + hours + minutes + seconds
    
    tHours = currentDate.hour/24
    tMinutes = currentDate.minute/(24*60)
    tSeconds = currentDate.second/(24*60*60)
    t_obs_preliminary = currentDate.toordinal()+tHours+tMinutes+tSeconds
    
    area_disk, bw_mask = imageMasks(imageName)                
     
    a = np.zeros_like(bw_mask)
   
    a[np.where(bw_mask>0)] = 1
    a = np.uint8(a)                
   
    mu, mu_rings = calc_mu_hmi(a)
   
    ndisk = np.count_nonzero(mu_rings)                
   
    for i in range(2,8):
        if i==5:
            bw1 = (bw_mask == 5) | (bw_mask == 6) | (bw_mask == 7)
            itemp = np.where(bw1)
        elif i == 6:
            bw1 = bw_mask == 6
            itemp = np.where(bw1)
        elif i == 7:
            bw1 = bw_mask == 7
            itemp = np.where(bw1)
        else:
            bw1 = bw_mask == i
            itemp = np.where(bw1)
   
        itemp = np.asarray(itemp)
       
        if itemp.shape[1]>0:
            aux_area[i-2] = itemp.shape[1]/area_disk
        else:
            aux_area[i-2] = 0
       
        for m in range(1,12):
            temp = bw1*(mu_rings==m)
            alpha_mu_preliminary = np.nansum(temp) / ndisk
            aux_alpha_mu_spot[i-2][m-1] = alpha_mu_preliminary

    area_c = np.append(area_c, [aux_area])

    alpha_mu_spot = np.append(alpha_mu_spot, [aux_alpha_mu_spot])
       
    time = np.append(time,t_obs_preliminary)

    aux_area = np.empty(6)
    aux_area.fill(np.nan)

    aux_alpha_mu_spot = np.empty([6,11])
    aux_alpha_mu_spot.fill(np.nan)

    area_c = area_c.reshape(1,6)
    alpha_mu_spot = alpha_mu_spot.reshape(1,6,11)

    
    print('\nArquivos salvos em '+partialOutputPath+' :\n')
    np.savetxt(partialOutputPath+'time.csv', time)
    print('time.csv\n')
    
    np.savetxt(partialOutputPath+'area_c.csv', area_c)
    print('area_c.csv\n')
    
    with open(partialOutputPath+'alpha_mu_spot.csv', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(alpha_mu_spot.shape))
    
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for dataSlice in alpha_mu_spot:
            outfile.write('# New slice\n')
            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
    
    print('alpha_mu_spot.csv\n')

def imageMasks(imageName):

    continuumImagePath = continuumPath + imageName + continuumSufix
    magImagePath = magPath + imageName + magSufix            
    
    bw_mask = np.zeros((1024,1024))
    
    bw_mask, area_disk = magMasks(magImagePath,bw_mask)

    
    bw_mask = continuumMasks(continuumImagePath,bw_mask)
    #print(bw_mask)

    fName = partialOutputPath+'bwMaskIMasks.csv'
    np.savetxt(fName, bw_mask)

    return area_disk, bw_mask            

def le_interp(t,x):

    y = x;

    ii = np.squeeze(np.where(~np.isnan(x)))
    ii1 = np.squeeze(np.where(np.isnan(x)))
    
    y[ii1] = interp1d(t[ii],x[ii], fill_value = 'extrapolate')(t[ii1])
    
    return y

def magMasks(magImagePath,bw_mask):
    
    currentImage = cv2.imread(magImagePath)
    grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grayImage,10,1,cv2.THRESH_BINARY)
    
    mu, mu_rings = calc_mu_hmi(thresh)
            
            
    area_disk = np.count_nonzero(mu>0)
            
    bw1 = grayImage>=(128+20)
    bw2 = grayImage<=(128-20)
    bw3 = bw1 | bw2
    bw4 = np.uint8(bw3 & (mu_rings > 0))
    #bw4 = (np.ma.masked_where(bw3, mu_rig_rings > 0).mask)
    
    #TESTAR DBSCAN 
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw4, connectivity=4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 10 
    bw5 = np.zeros((output.shape))

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            bw5[output == i + 1] = 1
    
    
    labels = label(bw5, connectivity=1)
    
    props = regionprops(labels)
    
    area = []
    
    for region in props:
        area.append(1e6*region.area/area_disk)

    area = np.asarray(area)

    th1_m = float(16.6883)
    th2_m = float(24.8066)
    th3_m = float(89.0673)
    
    bw_small = np.zeros(bw5.shape)
    ii = np.where(area <= th1_m)
    ii = np.array(ii)[0]
    flag_small = 0        
    if ii.size > 0:
        flag_small = 1
        for i in range (ii.size):
            x = props[ii[i]]['coords'][:,0]
            y = props[ii[i]]['coords'][:,1]
            bw_small[x,y] = 1 
            
    bw_media1 = np.zeros(bw5.shape)            
    ii = np.where((area > th1_m) & (area <= th2_m))
    ii = np.array(ii)[0]
    flag_media1 = 0        
    if ii.size > 0:
        flag_media1 = 1
        for i in range (ii.size):
            x = props[ii[i]]['coords'][:,0]
            y = props[ii[i]]['coords'][:,1]
            bw_media1[x,y] = 1 
    
    bw_media2 = np.zeros(bw5.shape)            
    ii = np.where((area > th2_m) & (area <= th3_m))
    ii = np.array(ii)[0]
    flag_media2 = 0        
    if ii.size > 0:
        flag_media2 = 1
        for i in range (ii.size):
            x = props[ii[i]]['coords'][:,0]
            y = props[ii[i]]['coords'][:,1]
            bw_media2[x,y] = 1 

    bw_large = np.zeros(bw5.shape)            
    ii = np.where(area > th3_m)
    ii = np.array(ii)[0]
    flag_large = 0        
    if ii.size > 0:
        flag_large = 1
        for i in range (ii.size):
            x = props[ii[i]]['coords'][:,0]
            y = props[ii[i]]['coords'][:,1]
            bw_large[x,y] = 1 


    
    bw_mask[np.where(mu_rings > 0)] = 1               
    
    if flag_small:
        bw_mask[np.where(bw_small==1)] = 2
    if flag_media1:
        bw_mask[np.where(bw_media1==1)] = 3
    if flag_media2:
        bw_mask[np.where(bw_media2==1)] = 4
    if flag_large:
        bw_mask[np.where(bw_large==1)] = 5

    return bw_mask, area_disk
   
def model_mdi_02_03(ds,dI):
    
    time, area_c, alpha_mu_spot = check_areas()
       
    alpha, t = prepareInput(time, alpha_mu_spot, dI)    

    F_f = np.squeeze(alpha[:,3,:] - alpha[:,4,:] - alpha[:,5,:])
    
    inputTime = t
    
    P = np.array([np.squeeze(alpha[:10,2,0]), 
                  np.squeeze(alpha[:10,3,0]), 
                  np.squeeze(alpha[:10,4,0]),
                  np.squeeze(alpha[:10,5,0])])

    P = P.reshape(P.shape[0]*P.shape[1])
        
    print('\nValue not scaled saved in '+outputPath+' :\n')
    
    PFileName = partialOutputPath+'P_unique.csv'
    np.savetxt(PFileName, P)
    print(PFileName+'\n')
    
    TimeFileName = partialOutputPath+'Time_unique.csv'
    np.savetxt(TimeFileName, inputTime)
    print(TimeFileName+'\n')

def prepareInput(time, inputData, dI):
    
    period = [dI]
    
    t = []
    alpha = np.zeros((11,6,len(period)))
    
    for j in range(len(period)):
        t1 = period[j]
        
        jj = np.where((time >= (t1 - dt/2)) & (time < (t1 + dt/2))) 
        
        for i in range(11):
            for k in range(6):
                temp = inputData[jj,k,i] #VERIFICAR SE OS ÍNDICES BATEM COM A ORDEM GRAVADA NO CHECK_AREAS()
                kk = np.where(np.isfinite(temp))
                alpha[i,k,j] = np.mean(temp[kk])
        
        t.append(t1)
        
    t = np.array(t)
    
    
    return alpha, t


def prepareOutput(time_tim, outputData):
    
    period = np.arange(dI,dF,dt)
    
    t = []
    output = []

    for j in range(len(period)):
        t1 = period[j]
        
        jj = np.where((time_tim >= (t1 - dt/2)) & (time_tim <= (t1 + dt/2))) 
        
        if (np.squeeze(jj)).size > 0:
            output.append(np.nanmean(outputData[jj]))
        else:
            output.append(np.nan)
        
        t.append(t1)
        
    t = np.array(t)
    output = np.array(output)
    # output = le_interp(period,output)
    
    return output, t


def TSI6HPrediction():
    
    np.random.seed(7)
    
    P1 = np.asarray(np.loadtxt(partialOutputPath+'P_unique.csv'))
    Time = np.asarray(np.loadtxt(partialOutputPath+'Time_unique.csv'))
    
    print(Time)
    P1 = P1.reshape(1,-1)
    
    P1.shape
    
    # Carregar
    with open(partialOutputPath+'scalerI.pkl', 'rb') as handle:
        scalerI = pickle.load(handle)
    
    #Salvar
    with open(partialOutputPath+'scalerO.pkl', 'rb') as handle:
        scalerO = pickle.load(handle)
        
    x1 = scalerI.transform(P1)
    
    units = 9
    loss = 'mean_squared_error'
    optimizer = 'adam'
    
    p = x1
    
    sx = p.reshape((p.shape[0], 1, p.shape[1])) 
       
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(sx.shape[1],sx.shape[2]), activation='tanh'))
    model.add(Dense(1))
    
    # checkpoint
    filepath = partialOutputPath+'weights.hdf5'
    
    #Load the best weights
    model.load_weights(filepath)
    
    #Compile model (required to make predictions)
    model.compile(loss=loss, optimizer=optimizer) #, metrics=['accuracy'])
    
    sy = model.predict(sx)
    
    y = scalerO.inverse_transform(sy)
    
    data = date.fromordinal(int(Time))
    horas = Time - int(Time)
    h = [0., 0.25, 0.5, 0.75]
    hrs = ['00','06','12','18']
    i = np.where(horas==h)
    j=hrs[np.squeeze(i)]
    tStamp = str(data)+' '+j+':00:00'
    
    output = str(np.squeeze(y))+','+tStamp

    tStamp = tStamp.replace(":", "")
    f = open(outputPath+tStamp+'.csv','a') 
    
    f.write(output + "\n")
    
    f.close

    return output
