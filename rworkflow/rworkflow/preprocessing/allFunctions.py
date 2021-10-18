from rworkflow.settings import *
from .library import *

def calc_mu_hmi(thresh):
        
    try:    
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
        
        #mu_rings = octave.interp1(ii0, ii1, mu, 'nearest')
        
        #mu_rings = interp1d(ii0, ii1, kind = 'nearest', fill_value = 'extrapolate')(mu)
 
        ''' As 10 linhas seguintes são a opção no caso de não utilizar o interpolador interp1.m do octave ou interp1d do Python.'''
        mu_rings = np.empty_like(mu)
        mu_rings[:] = np.nan
        
        limites = [1, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.075]

        for i in range(len(limites)-1):
            mu_rings[np.where((mu>limites[i+1]) & (mu<=limites[i]))] = ii1[i]
        
        mu_rings[np.where((mu>=0) & (mu<limites[10]))] = ii1[10]
        
        #'''
       
        return mu, mu_rings
    
    except:
        print('Error in the calc_mu_rings function')
        return (None, None)


def check_areas():
    
    #partialOutputPath= path + 'partial_output_files\\'

    time = np.loadtxt(partialOutputPath+'time.csv')
    area_c = np.loadtxt(partialOutputPath+'area_c.csv')
    alpha_mu_spot = np.loadtxt(partialOutputPath+'alpha_mu_spot.csv')
    #alpha_mu_spot = np.loadtxt(path+'FileName.txt')
    
    alpha_mu_spot = np.reshape(alpha_mu_spot,(np.size(time),6,11))
    
    #area_c = area_c.reshape([np.size(time),6]) # UMA OPÇÃO AO IF QUE VEM EM SEGUIDA, NECESSÁRIO PARA SOMAR OS VALORES NO CASO DA AREA_C SER UMA MATRIZ UNIDIMENSIONAL
    
    if area_c.ndim > 1:
        count = np.nansum(area_c,1)
    else:
        count = np.nansum(area_c)
    
    mu = np.nanmean(count)
    sigma = np.nanstd(count)
    
    n=count.size
    
    meanMat = matlib.repmat(mu, n, 1)
    sigmaMat = matlib.repmat(sigma, n, 1)
    
    count = count.reshape(meanMat.shape)
    
    outliers = np.abs(count - meanMat) > (3*sigmaMat)
    
    area_c[np.any(outliers,1),:] = np.nan
    
    alpha_mu_spot[np.any(outliers,1),:] = np.nan
    
    
    ck, kk = np.unique(time, return_index=True)
    
  
    n = area_c.shape[1]
    
    for j in range(n):
        area_c[kk,j] = le_interp(time[kk],area_c[kk,j])
    
    
    n = alpha_mu_spot.shape
    
    for j in range(n[1]):
        for i in range(n[2]):
            alpha_mu_spot[kk,j,i] = le_interp(time[kk],alpha_mu_spot[kk,j,i])
        
    print('\nFiles saved in '+partialOutputPath+':\n')
    np.savetxt(partialOutputPath+'check_areas_time_PY.csv', time)
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
     
    try:
        currentImage = cv2.imread(continuumImagePath)
    except FileNotFoundError:
        print("continuumMasks: Imagem " + continuumImagePath + "não encontrada!")

    try:    
        grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(grayImage,10,1,cv2.THRESH_BINARY)

        # figContinuum=plt.imshow(cv2.cvtColor(currentImage, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # figContinuum.figure.savefig(figuresPath+continuumImagePath[44:-4]+'_Continuum.png',dpi=300,format='png')
        
        # figContinuumGray = plt.imshow(grayImage, cmap="Greys")
        # plt.axis("off")
        # figContinuumGray.figure.savefig(figuresPath+continuumImagePath[44:-4]+'_ContinuumGray.png',dpi=300,format='png')
        
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
        
        '''TESTAR NOVAMENTE A INFORMAÇÃO ABAIX0:
        NO MATLAB, PARA A IMAGEM DE 25/09/2019, 00:00:00 (20190925_000000_Ic_flat_1k.jpg), 
        NÃO "SOBRA" NENHUM TRUE APÓS O IMCLEARBORDER. JÁ NO PYTHON, "SOBRAM" 1024 TRUES APÓS O CLEAR_BORDER.'''
        
        #bw2 = bw.astype(int)
        
        bw_mask[np.where(bw1)] = 6
        bw_mask[np.where(bw2)] = 7
        
        # x = np.arange(0,1024)
        # y = np.arange(1024,0,-1)
        
        # colors = ["#a0a0a0",  "#000000"]
        # # colors = ["#DF915E", "#C1524C", "#104210"]
        # cmap= clrs.ListedColormap(colors)
        # cmap.set_under("#FEEBA0")
        # # cmap.set_over("w")
        # norm= clrs.Normalize(vmin=0,vmax=7)
        
        # fig, ax = plt.subplots(figsize=(8,8))
        # plt.axis("off")
        # im = ax.contourf(x,y,bw_mask, levels=[0,6,7],
        #                  extend='both',cmap=cmap, norm=norm)
        # proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
        #          for pc in im.collections]

        # # plt.title('Identification of regions in magnetogram and their classification according to area size.',fontsize=14)
        # plt.legend(proxy, ['Solar disk','Penumbra','Umbra'], 
        #            loc=7, ncol=1, fontsize = 'medium', labelspacing=1,
        #            bbox_to_anchor=(1.25, 0.5))# 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} )
        # # fig.colorbar(im, extend="both")
        # # cbar = fig.colorbar(im, extend="both")
        # # cbar.ax.set_yticklabels(['Out of the disk', 'Disk background', 'Small regions', 'Medium regions', 'Medium regions 2', 'Large regions'])            
        
        # # plt.savefig(figuresPath+continuumImagePath[44:-4]+'_ContinuumClassified.pdf',
        # #             dpi=300,format='pdf',bbox_inches = "tight")
        
        # plt.savefig(figuresPath+continuumImagePath[44:-4]+'_teste.png',
        #             dpi=300,format='png',bbox_inches = "tight")

        # plt.show()

        return bw_mask
    
    except:
        print("continuumMasks failed: Erro na identificação e classificação de umbras e penumbras na imagem contínua " + continuumImagePath) 

def downloadImages(sufixo, path):
    
    currentDate = initialDate    
    
    expected=0
    downloaded=0
    
    logFileName = path+'logDownload.txt'
    log = open(logFileName,'w')
    #log.write('Files missing for download:\n')

    while currentDate<=finalDate:
        
        try:
            #BAIXANDO MAGNETOGRAMAS
            imageName = currentDate.strftime('%Y') + currentDate.strftime('%m') + currentDate.strftime('%d') + '_' + currentDate.strftime('%H') + currentDate.strftime('%M') + currentDate.strftime('%S') + sufixo
            imagePath = path + imageName
        
            if not(os.path.isfile(imagePath)):
                urlData = url + '/' + currentDate.strftime('%Y') + '/' + currentDate.strftime('%m') + '/' + currentDate.strftime('%d') + '/' + imageName
                image = request.urlopen(urlData)
                f = open(imagePath, 'wb')
                f.write(image.read())
                f.close()
                print(imagePath + '- arquivo baixado!')
            else:
                print(imagePath + '- arquivo já existente!')
            
            downloaded+=1
            # LENDO A IMAGEM:
            #img=mpimg.imread(imagePath)
            #imgplot = plt.imshow(img)
            #plt.show()
                       
        #except request.URLError:
        #    print(currentDate, "não disponível. (Erro de URL)")

        except:
            print("Ocorreu algum erro ao baixar ", imageName, ".")
            log.write(imageName+'\n')
            
        
        currentDate = currentDate + timedelta(minutes=resolution)  
        expected+=1

    log.close()
    
    '''
    data_atual = date.today()
    print(data_atual.day)
    print(data_atual.month)
    print(data_atual.year)
    print(data_atual.strftime('%d'))
    '''
    print('\nBaixados(ou existentes):'+str(downloaded)+'\tPrevistos:'+str(expected))
    print('Arquivos faltantes listados em: '+logFileName)


def fillsTSIGaps(fileName):
    
    #fF = open('outputPath'+fileName,'w')
    tsi = []
    
    with open(fileName,'r') as f:
        for line in f:
            tsi.append(float(line))    
    
    tsi = np.asarray(tsi)
    
    for i in range(len(tsi)):
        j=i
        nans = 0
        while(np.isnan(tsi[j])):
            nans+=1
            j+=1
        
        if nans<=4 and nans>0:
            newTsi = np.mean([tsi[i-1],tsi[j+1]])
            tsi[i:j] = newTsi
        elif nans>4:
            print('Amount of NaN above the supported!')
            break
        i=j
                   
    f.close()    
    fTsi = fileName
    np.savetxt(fTsi,tsi)
    
    return tsi


def geraAreas():

    # Counting the number of days will be processed    
    qtyDays = abs((finalDate - initialDate).days)+1
    print(str(qtyDays) + ' days to be processed...')
    
    currentDate = initialDate
    
    k=0
    
    aux_area = np.empty(6)
    aux_area.fill(np.nan)
    
    area_c = []
    #area_c = np.append(area_c,aux_area)
    
    aux_alpha_mu_spot = np.empty([6,11])
    aux_alpha_mu_spot.fill(np.nan)
    
    alpha_mu_spot = []
    #alpha_mu_spot = np.append(alpha_mu_spot,aux_alpha_mu_spot)
    
    time = []
    
    while currentDate<=finalDate:
        
        year = currentDate.strftime('%Y')
        month = currentDate.strftime('%m')
        day = currentDate.strftime('%d')
        hours = currentDate.strftime('%H')
        minutes = currentDate.strftime('%M')
        seconds = currentDate.strftime('%S')
    
        if ((hours=='00') | (hours == '06') | (hours == '12') | (hours == '18')) & ((minutes == '00') | (minutes == '30')):

            print('\nProcessing day '+str(currentDate))
    
            imageName = year + month + day + '_' + hours + minutes + seconds
            
            try:
                tHours = currentDate.hour/24
                tMinutes = currentDate.minute/(24*60)
                tSeconds = currentDate.second/(24*60*60)
                t_obs_preliminary = currentDate.toordinal()+tHours+tMinutes+tSeconds
            
                area_disk, bw_mask = imageMasks(imageName)                
                # x = np.arange(0,1024)
                # y = np.arange(1024,0,-1)
                # colors = ["#feeba0", "#ffab40", "#c5a483", "#2b1f12","#785225", "#a0a0a0",  "#000000"]
                # # colors = ["#feeba0", "#ffab40", "#f2a953", "#e3a764", "#d5a674", "#c5a483", "#b3a292", "#a0a0a0"]
                # cmap= clrs.ListedColormap(colors)
                # cmap.set_under("white")
                # #cmap.set_over("w")
                # norm= clrs.Normalize(vmin=0,vmax=7)
                
                # fig, ax = plt.subplots(figsize=(8,8))
                # plt.axis("off")
                # im = ax.contourf(x,y,bw_mask, levels=[0,1,2,3,4,5,6,7],
                #                   extend='both',cmap=cmap, norm=norm)
                # proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
                #           for pc in im.collections]
        
                # plt.legend(proxy, ['Off-disk','Solar disk','Small', 'Medium I', 
                #                     'Medium II', 'Large','Penumbra','Umbra'], 
                #             loc=7, ncol=1, fontsize = 'medium', labelspacing=1,
                #             bbox_to_anchor=(1.25, 0.5))# 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} )
                
                # plt.savefig(figuresPath+continuumImagePath[44:-4]+'Classified.png',
                #             dpi=300,format='png',bbox_inches = "tight")
                
        
                # plt.show()

                if k == 0:
                    a = np.zeros_like(bw_mask)
                    
                    a[np.where(bw_mask>0)] = 1
                    a = np.uint8(a)                
                    
                    mu, mu_rings = calc_mu_hmi(a)
                    
                    #ret,thresh = cv2.threshold(bw_mask,0,1,cv2.THRESH_BINARY)
                    #thresh = np.uint8(thresh)
                    #mu, mu_rings = calc_mu_hmi(thresh)
                    
         #           area_c = area_c.reshape(1,6)
         #           alpha_mu_spot = alpha_mu_spot.reshape(1,6,11)
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
                    
                    '''VALORES DE MU_RINGS APRESENTAM DIFERENÇAS ENTRE MATLAB E PYTHON'''
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
                
                k = k+1
            
                area_c = area_c.reshape(k,6)
                alpha_mu_spot = alpha_mu_spot.reshape(k,6,11)

            except FileNotFoundError:
                print('Masks not defined for ' + str(imageName))    
            
        currentDate = currentDate + timedelta(minutes=resolution)
    
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
    #return alpha_mu_spot,area_c,time

def imageMasks(imageName):

    #continuumImageName = imageName + continuumSufix
    #magImageName = imageName + magSufix
    
    continuumImagePath = continuumPath + imageName + continuumSufix
    magImagePath = magPath + imageName + magSufix            
    
    bw_mask = np.zeros((1024,1024))
    
    try:
        bw_mask, area_disk = magMasks(magImagePath,bw_mask)
    except FileNotFoundError:
        print('magMasks.py failed to ' + str(imageName))
        
    try:
        bw_mask = continuumMasks(continuumImagePath,bw_mask)
        #print(bw_mask)
    except FileNotFoundError:
        print('continuumMasks.py failed to ' + str(imageName))
  
    return area_disk, bw_mask            

def le_interp(t,x):

    y = x;

    ii = np.squeeze(np.where(~np.isnan(x)))
    ii1 = np.squeeze(np.where(np.isnan(x)))
    
    y[ii1] = interp1d(t[ii],x[ii], fill_value = 'extrapolate')(t[ii1])
    
    return y

def lookBack(x,y,n):
    if n>0:
        xAux = []
        yAux = []
        for i in range(n,x.shape[0]+1):
            xAux.append(x[i-n:i,:])
            yAux.append(y[i-1])
    else:
        xAux = x
        yAux = y
    
    return xAux, yAux
    

def magMasks(magImagePath,bw_mask):
    
    try:
        currentImage = cv2.imread(magImagePath)
        try:    
            grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(grayImage,10,1,cv2.THRESH_BINARY)
            
            # figMag=plt.imshow(currentImage, cmap="Greys")
            # plt.axis("off")
            # figMag.figure.savefig(figuresPath+magImagePath[38:-4]+'_Mag.png',dpi=300,format='png')
            # plt.show()
            
            # figMagGray = plt.imshow(grayImage, cmap="Greys")
            # plt.axis("off")
            # figMagGray.figure.savefig(figuresPath+magImagePath[38:-4]+'_MagGray.png',dpi=300,format='png')
            # plt.show()
            # '''MU_RINGS ESTÁ VINDO SEM NAN, DIFERENTE DO QUE ACONTECE NO MATLAB, O QUE ESTÁ DESTOANDO OS VALORES CALCULADOS PARA O BW_MASK, EM RELAÇÃO AOS CALCULADOS NO MATLAB'''
            mu, mu_rings = calc_mu_hmi(thresh)
            
            # figMu = plt.imshow(mu, cmap="Greys")
            # plt.axis("off")
            
            # figMu.figure.savefig(figuresPath+magImagePath[38:-4]+'_Mu.png',dpi=300,format='png')            

            # figMu_rings = plt.imshow(mu_rings, cmap="Greys")
            # plt.axis("off")
            
            # figMu_rings.figure.savefig(figuresPath+magImagePath[38:-4]+'_Mu_Rings.png',dpi=300,format='png')            
            
            area_disk = np.count_nonzero(mu>0)
    
        except:
            print("magMasks: Error in the cleaning and solar disk and ring setting steps" + magImagePath)  
        
        try:
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
            #area_total.append(area) #apenas uma verificação das áreas totais calculadas
    
            '''VERIFICAR SE NA LINHA ABAIXO DEVE REALMENTE SER EXCLUÍDA A PRIMEIRA ÁREA (VALOR MÁXIMO), CUJO VALOR NÃO CONDIZ COM OS RESULTADOS EM MATLAB (em alguns casos!!!)'''
            #area = np.asarray(area[1:])
            #props = props[1:]
            area = np.asarray(area)
        except:
            print("magMasks: Error in the image pre-classification step " + magImagePath)  
            
        try:  
            '''ESSES VALORES DE THRESHOLDS DEVEM FICAR EM UM ARQUIVO MESMO?? COMO SÃO ESTIMADOS???'''
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
    
    #         x = np.arange(0,1024)
    #         y = np.arange(1024,0,-1)
            
    #         colors = ["#feeba0", "#ffab40", "#c5a483", "#2b1f12","#785225"]
    #         # colors = ["#FEEBA0", "#104210", "#618699", "#DF915E", "#C1524C"]
    #         cmap= clrs.ListedColormap(colors)
    #         cmap.set_under("white")
    #         cmap.set_over("w")
    #         norm= clrs.Normalize(vmin=0,vmax=5)
            
    #         fig, ax = plt.subplots(figsize=(8,8))
    #         plt.axis("off")
    #         im = ax.contourf(x,y,bw_mask, levels=[0,1,2,3,4,5],
    #                           extend='both',cmap=cmap, norm=norm)
    #         proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
    # for pc in im.collections]

    #         # plt.title('Identification of regions in magnetogram and their classification according to area size.',fontsize=14)
    #         plt.legend(proxy, ['Off-disk','Solar disk','Small', 'Medium I', 
    #                             'Medium II', 'Large'], 
    #                     loc=7, ncol=1, fontsize = 'medium', labelspacing=1,
    #                     bbox_to_anchor=(1.2, 0.5))# 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} )
    #         # fig.colorbar(im, extend="both")
    #         # cbar = fig.colorbar(im, extend="both")
    #         # cbar.ax.set_yticklabels(['Out of the disk', 'Disk background', 'Small regions', 'Medium regions', 'Medium regions 2', 'Large regions'])            
    #         plt.show()
            
    #         fig.savefig(figuresPath+magImagePath[38:-4]+'_MagClassified.png',
    #                     dpi=300,format='png',bbox_inches = "tight")

            
            return bw_mask, area_disk
        
        except:
            print("magMasks: Error in the image classification process " + magImagePath)
        
    except FileNotFoundError:
        print("magMasks failed: Image " + magImagePath + "not found!")


def model_eve_01_01(ds):
    time, area_c, alpha_mu_spot = check_areas()
    
    #dt = 1
    
    alpha, t = prepareInput(time,alpha_mu_spot)
    
    time_eve, irrad,  wav = read_eve_data()
    
    period = np.arange(dI,dF,dt)
    
    lines = []
    #for i in range(wav.size):
    for k in range(len(SSILines)):
        
        lines.append(list(np.where(wav==SSILines[k])[0]))
        
        i = np.squeeze(lines[k][0])
        
        print('Computing: ',wav[i],'nm \n')
        #clear P T chi2 irrad_sorce_t
        
        #irrad_eve_t = prepareSSIOutput(time_eve, irrad, i, ds) #IMPLEMENTAR!!
        irrad_eve_t = []
        
        for j in range(len(period)):
            t1 = period[j]
            #std1 = np.nanstd(irrad[:,i],ddof=1)
            
            jj = np.where((time_eve >= (t1 - dt/2)) & (time_eve < (t1 + dt/2)))
            #print(jj),
            if len(jj) > 0:
                irrad_eve_t.append(np.nanmedian(irrad[jj,i]))
            else:
                irrad_eve_t.append(np.nan)
            #print(irrad_eve_t[j])
    
        irrad_eve_t = np.asarray(irrad_eve_t)
        irrad_eve_t = le_interp(period,irrad_eve_t)
        
        #kk = np.where((period >= date.toordinal(initialDate)) & (period <= time_eve[-1]) )
        kk = np.where((period >= dI) & (period <= dF) )
        kk = np.squeeze(kk)
        
        #ds = 1
        
        T = irrad_eve_t[kk[ds:]]
                      
        #nalpha = alpha[:10,1:,kk[:len(kk)-(ds)]]
        
        #P = nalpha.reshape(nalpha.shape[0]*nalpha.shape[1],nalpha.shape[2])
        
        #P = nalpha.reshape((nalpha.shape[0]*nalpha.shape[1], nalpha.shape[2]), order='F')
        
        P = np.array([np.squeeze(alpha[:10,2,kk[:len(kk)-(ds)]]), 
                      np.squeeze(alpha[:10,3,kk[:len(kk)-(ds)]]), 
                      np.squeeze(alpha[:10,4,kk[:len(kk)-(ds)]]),
                      np.squeeze(alpha[:10,5,kk[:len(kk)-(ds)]])])

        P = P.reshape(P.shape[0]*P.shape[1],len(t)-ds)
        
        Time = t[kk[:len(kk)-(ds)]]
        
        rLabel = str(int(dt*ds))+'days.csv'

        print('\nMatrizes não normalizadas salvas em '+outputPath+' :\n')
        PFName = outputPath+'P_'+str(k+1)+'_'+outputLabel+rLabel
        #np.savetxt(outputPath+str(wav[i])+'_P_EVE_PY.csv', P)
        np.savetxt(PFName, P)
        
        #print(outputPath+'P_PY'+str(k+1)+'.csv')
        print(PFName)
        
        TFName = outputPath+'T_'+str(k+1)+'_'+outputLabel+rLabel
        
        #np.savetxt(outputPath+str(wav[i])+'_T_EVE_PY.csv', T)
        np.savetxt(TFName, T)
       
        #print(str(wav[i])+'_T_EVE_PY.csv\n')
        print(TFName)
        
        TimeFName = outputPath+'Time_'+str(k+1)+'_'+outputLabel+rLabel
        
        #np.savetxt(outputPath+'Eve_Time_PY.csv', Time)
        print(TimeFName)
        
    
def model_mdi_02_03(ds):
    
    
    time, area_c, alpha_mu_spot = check_areas()
       
    time_tim, tsi_tim = read_tim_tsi()
    
    alpha, t = prepareInput(time, alpha_mu_spot)    
    #alpha, t = prepareInput2(time, alpha_mu_spot)    
    
    tsi, tsi_t = prepareOutput(time_tim, tsi_tim)
    #tsi, tsi_t = prepareOutput2(time_tim, tsi_tim)
    '''???'''
    F_f = np.squeeze(alpha[:,3,:] - alpha[:,4,:] - alpha[:,5,:])
    
    '''QUE DATA É ESSA??? '''
    #kk = np.array(np.where((t >= date.toordinal(initialDate)) & np.isfinite(F_f[0,:]))) #COmentado por Amita em 17/11/2020 pois deu erro no reshape do P
    kk = np.array(np.where((t >= dI) & (np.isfinite(F_f[0,:])) & (t < dF)))
    kk = np.squeeze(kk)
    
    #ds = 1 #prediction window. For instante, 1 = dt*24 hours prediction, 2 = 2*dt hours prediction
    
    inputTime = t[kk[:len(kk)-(ds)]]
    outputTime = t[kk[ds:]]
    
    T = tsi[kk[ds:]]
    
      
    P = np.array([np.squeeze(alpha[:10,2,kk[:len(kk)-(ds)]]), 
                  np.squeeze(alpha[:10,3,kk[:len(kk)-(ds)]]), 
                  np.squeeze(alpha[:10,4,kk[:len(kk)-(ds)]]),
                  np.squeeze(alpha[:10,5,kk[:len(kk)-(ds)]])])

#    with open(partialOutputPath+'\\P_PY3D.csv', 'w') as outfile:
#        outfile.write('# Array shape: {0}\n'.format(P.shape))
#
#        slice=0
#        #print('Saving slices: '),
#        for dataSlice in P:
#            outfile.write('# New slice\n')
#            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
#            slice+=1
#            #print(str(slice)+'\t'),
#        
#    print('Arquivo salvo: '+partialOutputPath+'\\P_PY3D.csv')
    
    P = P.reshape(P.shape[0]*P.shape[1],len(t)-ds)
    T = T.transpose()
    
    #nans = np.isnan(T)
    #x = lambda z: z.nonzero()[0]
    #T[nans] = np.interp(x(nans), x(~nans), T[~nans])

    #P = np.transpose(P)
    
    if dt<1:
        rLabel = str(int((24*dt)*ds))+'hours.csv'
    elif dt==1:
        rLabel = str(int(dt*ds))+'days.csv'

    print('\nMatrizes não normalizadas salvas em '+outputPath+' :\n')
    
    #PFileName = 'P_'+HoursOfResolution+'hours.csv'
    PFileName = 'P_'+outputLabel+'_'+rLabel
    np.savetxt(outputPath+PFileName, P)
    print(PFileName+'\n')
    
    #TFileName = 'T_'+HoursOfResolution+'hours.csv'
    TFileName = 'T_'+outputLabel+'_'+rLabel
    np.savetxt(outputPath+TFileName, T)
    print(TFileName+'\n')
    
    #TimeFileName = 'Time_'+HoursOfResolution+'hours.csv'
    TimeFileName = 'Time_'+outputLabel+'_'+rLabel
    np.savetxt(outputPath+TimeFileName, inputTime)
    print(TimeFileName+'\n')
    
    
    #rnn(P,T)

def preencheLacunas(qtdDelta15):
    
    url = "http://jsoc.stanford.edu/data/hmi/images"
    #f = open(path+'\\substitutos\\logDownload.txt','r')
    f = open(path+'MissingImages_6hours.txt','r')
    resolvidos = 0
    faltantes = 0
    lines = 0
    
    pathSubstituicao = path+'Substitutos\\'
    renamedPath = pathSubstituicao+'Renomeados\\'
    
    log = open(pathSubstituicao+'logPreenchimento.txt','w')
    
    for linha in f:
        
        lines+=1
        year = linha[0:4]
        month = linha[4:6]
        day = linha[6:8]
        hours = linha[9:11]
        minutes = linha[11:13]
        sufix = linha[15:-1]
        
        #print(sufix)
        data = day+'/'+month+'/'+year+' '+hours+':'+minutes+':00'
        data = datetime.strptime(data,'%d/%m/%Y %H:%M:%S')
    
        delta = 15
        changes = 0
        done = 0   
        
        originalImageName = data.strftime('%Y') + data.strftime('%m') + data.strftime('%d') + '_' + data.strftime('%H') + data.strftime('%M') + data.strftime('%S') + sufix
        renamedImagePath = renamedPath + originalImageName
        
        log.write('Opções para o arquivo '+str(data)+'\n')
        print('\n\nOpções para '+linha)
        
        
        while (changes<qtdDelta15) and (~done):    

            fail = 0
            currentDate = data + timedelta(minutes=delta)
            #log.write(str(currentDate)+'\n')
            imageName = currentDate.strftime('%Y') + currentDate.strftime('%m') + currentDate.strftime('%d') + '_' + currentDate.strftime('%H') + currentDate.strftime('%M') + currentDate.strftime('%S') + sufix
            imagePath = pathSubstituicao + imageName
    
            urlData = url + '/' + currentDate.strftime('%Y') + '/' + currentDate.strftime('%m') + '/' + currentDate.strftime('%d') + '/' + imageName
            #print('Verificando disponibilidade À FRENTE.')
            try:
                if (not(os.path.isfile(renamedImagePath))):
                    #print('Tentando baixar em '+urlData)                    
                    image = request.urlopen(urlData)
                    #print(imageName+' baixado')
                    #print('Tentando salvar em: '+imagePath)
                    im = image.read()
                    fo = open(imagePath, 'wb')
                    fo.write(im)
                    fo.close()
                    fr = open(renamedImagePath, 'wb')
                    fr.write(im)
                    fr.close()
                    
                    print(imageName + '- arquivo baixado!')
                #else:
                    #print(imagePath + '- arquivo já existente!')
                done = 1
                log.write('Substituído por '+imageName+'\n\n')
                #print(imageName+' OK')
                print('Sugestão para substituição em instante posterior baixada!')
                break
            except:
                fail = 1
                #print('FAILED')
                

            if fail:
                delta *= -1    
                currentDate = data + timedelta(minutes=delta)
                imageName = currentDate.strftime('%Y') + currentDate.strftime('%m') + currentDate.strftime('%d') + '_' + currentDate.strftime('%H') + currentDate.strftime('%M') + currentDate.strftime('%S') + sufix
                imagePath = pathSubstituicao + imageName
                
                urlData = url + '/' + currentDate.strftime('%Y') + '/' + currentDate.strftime('%m') + '/' + currentDate.strftime('%d') + '/' + imageName
                #print('Verificando disponibilidade ATRÁS.')
                try:
                    if (not(os.path.isfile(renamedImagePath))):
                        #print('Tentando baixar em '+urlData)                    
                        image = request.urlopen(urlData)
                        #print(imageName+' baixado')
                        imagePath = pathSubstituicao + imageName
                        #print('Tentando salvar em: '+imagePath)
                        im = image.read()
                        fo = open(imagePath, 'wb')
                        fo.write(im)
                        fo.close()
                        fr = open(renamedImagePath, 'wb')
                        fr.write(im)
                        fr.close()
                        
                        print(imageName + '- arquivo baixado!')
                    #else:
                        #print(imagePath + '- arquivo já existente!')
                    done = 1
                    log.write('Substituído por '+imageName+'\n\n')
                    #print(imageName+' OK')
                    print('Sugestão para substituição em instante anterior baixada!')
                    break
                except:
                    #print(imageName+' não baixado\n')
                    #print('FAILED')
                    print('',end='')
                    
                    
                delta *= -1
                
            changes+=1
            delta+=15

        #print('Arquivo '+linha)

        if done:
            resolvidos+=1
        #    print('RESOLVIDO!\n')
        else:
            faltantes+=1
        #    print('AINDA FALTA!\n')
    f.close()
    log.close()
    print('\n\nTOTAL DE ARQUIVOS: '+str(lines))
    print('RESOLVIDOS: '+str(resolvidos))
    print('FALTANTES: '+str(faltantes))

def prepareInput(time, inputData):
    
    period = np.arange(dI,dF,dt)
    
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
    
    for i in range(11):
        for k in range(6):
            alpha[i,k,0] = alpha[i,k,1]
    
    for i in range(11):
        for k in range(6):
            alpha[i,k,:] = le_interp(t, np.squeeze(alpha[i,k,:]))
    
    return alpha, t


def prepareOutput(time_tim, outputData):
    
    period = np.arange(dI,dF,dt)
    
    t = []
    output = []

    for j in range(len(period)):
        t1 = period[j]
        
        jj = np.where((time_tim >= (t1 - dt/2)) & (time_tim < (t1 + dt/2))) 
        
        if (np.squeeze(jj)).size > 0:
            output.append(np.nanmean(outputData[jj]))
        else:
            output.append(np.nan)
        
        t.append(t1)
        
    t = np.array(t)
    output = np.array(output)
    output = le_interp(period,output)
    
    return output, t


def read_tim_tsi():
    
    #tsiFile = 'sorce_tsi_L3_c06h_latest.txt'
    
    #urlTsi = 'http://lasp.colorado.edu/data/sorce/tsi_data/six_hourly/' + tsiFile
   
    filePath = path+tsiFile
    
    #uploadedFile = request.urlopen(url)

    #f = open(filePath, 'wb')

    #f.write(uploadedFile.read())
    #f.close()

    

    with open(filePath,'r') as f: 
        #read only data, ignore headers 
        lines = f.readlines()[134:] 
        # create the arrays 
        data = ''
        time_tim = np.zeros(len(lines))

        tsi_tim = np.zeros(len(lines))

        tsi_tim_sig = np.zeros(len(lines))
        
        # convert strings to floats and put into arrays 
        for i in range(len(lines)):
            data, data2, data3, data3, tsi_tim[i], data4, data5, data6, tsi_tim_sig[i], seinao, seinao, seinao, seinao, seinao, seinao= lines[i].split()
            
            yyyy = int(data[0:4])
            mm = int(data[4:6])
            ddf = float(data[6:])
            dd = int(ddf)
            hh = ddf - dd
            
            time_tim[i] = date.toordinal(datetime(yyyy,mm,dd))+hh
            
            tsi_tim[i] = float(tsi_tim[i])
            tsi_tim_sig[i] = float(tsi_tim_sig[i])
    
    tsi_tim[np.where(tsi_tim == 0)] = np.nan
    tsi_tim_sig[np.where(tsi_tim_sig == 0)] = np.nan

    
    n1 = len(time_tim)
    n2 = len(tsi_tim)
    
    if np.not_equal(n1,n2):
        n = min(n1,n2)
        tsi_tim = tsi_tim[:n]
        time_tim = time_tim[:n]
        tsi_tim_sig = tsi_tim_sig[:n]
    
    return time_tim, tsi_tim #, tsi_tim_sig

