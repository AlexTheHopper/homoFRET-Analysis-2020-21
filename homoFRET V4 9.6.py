import math
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
import scipy
import statistics as stats
import animatplot as amp
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf



###This .py was written for Alexander Hopper's Masters of Science - Physics at The University of Melbourne in 2020-2021.###
###It allows the computation and interpretation of .tif files from Olympus microscopes for anisotropy analysis###
###Outputs include time-averaged anisotropy maps, anisotropy S.D. maps, anisotropy GIFs etc.###
###It wa then further expanded upon during a research assistance role, also under Dr. Elizabeth Hinde at the University of Melbourne.###
###This project aimed to develop the pipelining for homoFRET - a molecular proximity based assay involving only one fluorescence protein.###
###It was carried out under the supervision of Dr. Elizabeth Hinde###

###Keep in mind this was basically how I learned Python and before I knew anything about OOP, every aspect of this can be impoved immensely###

#Open channels
#TIF File name (only enter Ch1):

fileCh1 = "90 70  H2B-eGFP HYPO cell 24 20x_Ch1.tif"

#1.1634 8x
#1.2519 20x

#Enter GPFactor here after flu callibration.
GPFactor = 0.7853


#################
#####TOGGLES#####
#################

#Set =1 when doing fluorescein callibration.
flu = 0

#Free eGFP(or other FP) anisotropy value to calculate delta r:
free_aniso = 0.4

#set to mask out low intensities,lowlim will be the lowest kept intensities. Set to 0 for no mask.
lowlim = 3.5

#Set =1 for FV3000 LASER and =0 for ISS LASER.
FV3000 = 1

#Set =1 for temporal analysis and =0 to average all frames for only spatial analysis.
time_analysis = 1

#Set =1 to export data to a .csv.
export = 0

#Set =1 to create an anisotropy pseudocolour map.
anisotropy_pseudocolour = 1

#Set =1 to show anisotropy traces based on intensity.
aniso_trace_intensity = 1

#Set =1 to calculate neighbouring anisotropy trace correlation
correlate = 1
quick_correlate = 1

#Set =1 to analyse standard deviation of anisotropy.
standard = 1
#Set =1 to create a standard deviation pseudocolour map.
stddev_pseudocolour = 1
lowlimstdev = 0.01

#Set =1 to analyse Fourier/Power spectrum data.
power = 1
lowpower = 1
highpower = 1

#Set =1 to output an anisotropy .gif.
gif = 0
#Set =1 to analyse total image FRET percentage temporally.
perovertime = 1

#Set =1 to analyse a specific ROI coordinate:
specific = 0
specific_x = 45
specific_y = 45
specific_list = []


####################
#####PARAMETERS#####
####################

#Set this to the acquisition time for one frame - for Fourier analysis.
frame_time = 0.48
#Usually 1.6s for 256x256 and 0.48s for 128x128.
nyquist = 1 /(2 * frame_time)

#frameavg is the number of averaged frames - reccomended == 20. Set == 0 for no rolling average.
frameavg = 5

#Fixes photobleaching trend - reccomended == 10. Set =0 to not do this.
fixnum = 0

#Set =1 to output delta r values and 0 to output anisotropy values
#IF delta_r =1, set iso_threshold to 0.1, and if delta_r =0, set iso_threshold to the untreated anisotropy.
delta_r = 0

#Thresholding intervals
#Pixels with anisotropy lower than iso_threshold will be considered FRET pixels.
iso_threshold = 0.3
#stdev_threshold is the lower limit for pseudocolouring standard deviation.
stdev_threshold = 0.006
#stdev_interval is for pseudocolouring the different levels of standard deviation.
stdev_interval = 0.003
#stdev_interval_num is how many bins to pseudocolour the stdev with.
stdev_interval_num = 4

#ALL functions used:
if True:
    NaN = np.nan
    def maximum(list):
        dummy = np.NaN
        location = np.NaN
        for i in range(len(list)):
            if math.isnan(dummy) == True:
                dummy = list[i]
            if math.isnan(dummy) == False and list[i] > dummy:
                dummy = list[i]
                location = i
        return [dummy,location-1]
    def minimum(list):
        dummy = np.NaN
        location = np.NaN
        for i in range(len(list)):
            if math.isnan(dummy) == True:
                dummy = list[i]
            if math.isnan(dummy) == False and list[i] < dummy:
                dummy = list[i]
                location = i
        return [dummy,location-1]
    def maximum2d(array):
        dummy = np.NaN
        location = [np.NaN,np.NaN]
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if math.isnan(dummy) == True:
                    dummy = array[i,j]
                if math.isnan(dummy) == False and array[i,j] > dummy:
                    dummy = array[i,j]
                    location = [i,j]
        return [dummy,location[0]-1,location[1]-1]
    def minimum2d(array):
        dummy = np.NaN
        location = [np.NaN,np.NaN]
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if math.isnan(dummy) == True:
                    dummy = array[i,j]
                if math.isnan(dummy) == False and array[i,j] < dummy:
                    dummy = array[i,j]
                    location = [i,j]
        return [dummy,location[0]-1,location[1]-1]
    def nanmean(array):
        mean = 0
        count = 0

        for i in range(len(array)):
            for j in range(len(array)):
                if math.isnan(array[i,j]) == False:
                    mean += array[i,j]
                    count += 1

        if count == 0:
            return(np.nan)
        return mean/count
    def divzero(array1,array2):
        result = np.zeros((len(array1),len(array1)))

        for i in range(len(array1)):
            for j in range(len(array1)):
                if array1[i,j] == 0 or array2[i,j] == 0:
                    result[i,j] = 'nan'
                else:
                    result[i,j] = array1[i,j]/array2[i,j]
        return(result)
    def moving_avg(array,avg,N):
        if avg == 0:
            print("No moving average for Ch",N)
            return(array)
        new0 = array.shape[0]-avg+1
        final = np.ones((new0,array.shape[1],array.shape[2]),)

        a = 100
        per = 0
        for k in range(new0):

            per = np.ceil(k / new0 * 10)
            if a != per:
                print("Moving Average:",per*10 / 2 + 50*(N-1),"%. (Ch",N,"/ 2)" )
            a = per


            for i in range(array.shape[1]):
                for j in range(array.shape[2]):

                    if k == 0:
                        element = 0
                        for n in range(avg):
                            element += array[k+n,i,j]
                        final[k,i,j] = element/avg

                    if k != 0:
                        final[k,i,j] = (final[k-1,i,j]*avg-array[k-1,i,j]+array[k+avg-1,i,j])/avg
        return(final)
    def moving_average_list(list, N):
        return np.convolve(list, np.ones(N), 'valid') / N
    def image(array,title,map):
        plt.imshow(array)
        plt.set_cmap(map)
        plt.colorbar()
        #plt.clim(0.3,0.4)
        plt.title(title)
        plt.show()
    def imagelim(array,title,map,low,high):
        plt.imshow(array)
        plt.set_cmap(map)
        plt.colorbar()
        plt.clim(low,high)
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='black')
        plt.title(title)
        plt.show()
    def threshold(array,lowlim):
        for i in range(len(array)):
            for j in range(len(array)):
                if array[i,j] < lowlim:
                    array[i,j] = 'nan'
        return array
    def process(CH_pp,CH_pl,GPFactor):

        CH_pp_gp = np.multiply(CH_pp,GPFactor)

        numerator = np.subtract(CH_pl,CH_pp_gp)
        NA_corrected = np.multiply(1.2,CH_pp_gp)
        denominator_NA_corrected = np.add(CH_pl,NA_corrected)
        GP = divzero(numerator,denominator_NA_corrected)
        return(GP)
    def avgfix(list,n):
        result = list[:]
        if n == 0:
            return(result)
        target = 0
        count = 1

        for i in range(n):
            target += list[i]
        target /= n

        for i in range(math.floor(len(list)/n)):
            avg = 0
            for j in range(n):
                avg += list[n*i+j]
            avg /= n
            avg = target - avg
            for j in range(n):
                result[n*i+j] += avg

        rest = len(list)%n
        avg = 0
        #print(rest)

        if rest != 0:
            for i in range(rest):
                avg += list[-(i+1)]
            avg /= rest
            avg = target - avg
            for i in range(rest):
                result[-(i+1)] += avg

        return result
    def percentage2d(array,threshold):
        count = 0
        total = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if math.isnan(array[i,j]) == False:
                    total += 1
                    if array[i,j] > threshold:
                        count += 1
        if total == 0:
            return
        return count/total
    def exporttocsv(list,name):
        file = open('homoFRET_export.csv', 'a+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows([name])
        file = open('homoFRET_export.csv', 'a+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows([list])
    def powerspectrum(list,interval):
        ps = np.abs(np.fft.fft(list))**2
        freqs = np.fft.fftfreq(len(list), interval)
        idx = np.argsort(freqs)
        return([freqs[idx],ps[idx]])


#Gets name of Ch2
fileCh2 = fileCh1[:-5] + '2' + fileCh1[-4:]


print("Reading:")
print(fileCh1)
print(fileCh2)
im1 = io.imread(fileCh1)
im2 = io.imread(fileCh2)
image1 = np.array(im1)
image2 = np.array(im2)



print("Image size:",im1.shape)

#Average anisotropy analysis
if time_analysis == 0 or flu == 1:

    #Defines image size for neatness
    if len(image1.shape) == 3:
        ims1 = image1.shape[1]
        ims2 = image1.shape[2]
    if len(image1.shape) == 2:
        ims1 = image1.shape[0]
        ims2 = image1.shape[1]



    #Averages all frames for no moving average so new channel images are CH1 and CH2.
    if len(image1.shape) == 3:

        CH1 = np.zeros((ims1,ims2))
        for i in range(ims1):
            for j in range(ims2):
                CH1[i,j] = image1[:,i,j].mean()

        CH2 = np.zeros((ims1,ims2))
        for i in range(ims1):
            for j in range(ims2):
                CH2[i,j] = image2[:,i,j].mean()
    if len(image1.shape) == 2:

        CH1 = image1
        CH2 = image2


    #Different LASERs:
    if FV3000 == 1:
        CH_pp = (CH1)
        CH_pl = (CH2)
    if FV3000 == 0:
        CH_pp = (CH2)
        CH_pl = (CH1)

    #Calculate GPFactor if flu == 1.
    if flu == 1:
        GPFactor = divzero(CH_pl,CH_pp)
        GPFactor = nanmean(GPFactor)
        print("GPFactor is: ",round(GPFactor,4))
        #Correct for detector sensitivities w/ GPFactor

    #For cell image analysis:
    if flu != 1:
        #We need to account for the intensity thresholding; we do this with check_array.
        #check_array is ones for good pixels and nan for bad pixels.
        check_array = np.ones((ims1,ims2))
        for i in range(ims1):
            for j in range(ims2):

                if CH1[i,j] < lowlim or CH2[i,j] < lowlim:
                    check_array[i,j] = NaN

        #gives a 2D image of average anisotropies.
        GP = process(CH_pp,CH_pl,GPFactor)
        GP = ndimage.median_filter(GP, size = 3)

        #MeanGP gives an average anisotropy value for the whole image (only good pixels)
        meanGP = (nanmean(np.multiply(GP,check_array)))
        print("meanGP: ",round(meanGP,4))

        if delta_r == 1:
            GP = np.subtract(free_aniso,GP)

        meanGP = (nanmean(np.multiply(GP,check_array)))

        if delta_r == 1:
            print("Average delta r: ",round(meanGP,4))
        if delta_r == 0:
            print("Average anisotropy: ",round(meanGP,4))
        # print(maximum2d(CH1),maximum2d(CH2))
        max_intensity = max(maximum2d(CH1)[0],maximum2d(CH2)[0])

        imagelim(CH1,"Intensity Ch1","gist_gray",0,max_intensity)
        imagelim(CH2,"Intensity Ch2","gist_gray",0,max_intensity)
        # print(maximum2d(np.multiply(GP,check_array)),minimum2d(np.multiply(GP,check_array)))
        if delta_r == 1:
            imagelim(np.multiply(GP,check_array),"Average delta r","jet",0.05,0.15)
        if delta_r == 0:
            imagelim(np.multiply(GP,check_array),"Average anisotropy","jet_r",0.2,0.4)


        #Anisotropy threshold pseudocolouring and percentage.
        if anisotropy_pseudocolour == 1:
            GP_pseudo = np.zeros((ims1,ims2))

            for i in range(ims1):
                for j in range(ims2):
                    if delta_r == 1:
                        if GP[i,j] > iso_threshold and math.isnan(check_array[i,j]) == False:
                            GP_pseudo[i,j] = 1
                    if delta_r == 0:
                        if GP[i,j] < iso_threshold and math.isnan(check_array[i,j]) == False:
                            GP_pseudo[i,j] = 1

            # print("total and count are",total,count)
            if delta_r == 1:
                print("Percentage by delta r pseudocolour is",100*round(percentage2d(np.multiply(GP_pseudo,check_array),0.5),4),"%")
                imagelim(np.multiply(GP_pseudo,check_array),"delta r pseudocolour","coolwarm",0,1)
            if delta_r == 0:
                print("Percentage by anisotropy pseudocolour is",100*round(percentage2d(np.multiply(GP_pseudo,check_array),0.5),4),"%")
                imagelim(np.multiply(GP_pseudo,check_array),"anisotropy pseudocolour","coolwarm",0,1)



        intensity_list = []
        anisotropy_list = []


        for i in range(ims1):
            for j in range(ims2):
                if math.isnan(check_array[i,j]) == False:

                    a = (CH1[i,j] + CH2[i,j]) / 2
                    intensity_list.append(a)
                    anisotropy_list.append(GP[i,j])

        if export == 1:

            file = open('homoFRET_export.csv', 'a+', newline ='')
            with file:
                write = csv.writer(file)
                write.writerows([[fileCh1[:-8]]])
            exporttocsv(intensity_list,["Intensity"])
            if delta_r == 1:
                exporttocsv(anisotropy_list,["delta r"])
            if delta_r == 0:
                exporttocsv(anisotropy_list,["anisotropy"])
            file = open('homoFRET_export.csv', 'a+', newline ='')
            with file:
                write = csv.writer(file)
                write.writerows([[""]])


            # plt.hist(anisotropy_list, density=False, bins=30)  # density=False would make counts
            # plt.ylabel('Counts')
            # plt.xlabel('Delta r');
            # plt.show()

#Fluctuation and standard deviation analysis
if time_analysis == 1 and flu == 0:
    if frameavg > im1.shape[0]:
        print("Moving average window too large or input too short!")

    #We first apply a moving average to both channel's data.
    print("Mov Avg on CH1")
    CH1 = moving_avg(im1,frameavg,1)

    # CH1 = image1

    print("Mov Avg on CH2")
    CH2 = moving_avg(im2,frameavg,2)

    #Prepares lists & array for anisotropy values both for whole image and for 2x2 ROI
    meanGPvalues = []
    GPvalues = np.ones((im1.shape[1],im1.shape[2]))


    #This array differentiates pixels based on the threshold intensity.
    #For temporal analysis, it must be 3D
    check_array = np.ones((CH1.shape[0],CH1.shape[1]-1,CH1.shape[2]-1))

    total_check = (CH1.shape[1]-1) * (CH1.shape[2]-1)

    #alliso is a 3D array through time of each 2x2 anisotropy value at all locations.
    alliso = np.zeros((CH1.shape[0],CH1.shape[1]-1,CH1.shape[2]-1))


    a = 100
    per = 0
    for n in range(CH1.shape[0]-1):
        per = np.ceil(n / CH1.shape[0] * 10)
        if a != per:
            print("Calculating delta r/anisotropy values:",per*10,"%")
            a = per
        for i in range(CH1.shape[1]-1):
            for j in range(CH1.shape[2]-1):

                Ch1mean = (CH1[n,i,j] + CH1[n,i+1,j] + CH1[n,i,j+1] + CH1[n,i+1,j+1]) / 4
                Ch2mean = (CH2[n,i,j] + CH2[n,i+1,j] + CH2[n,i,j+1] + CH2[n,i+1,j+1]) / 4

                if Ch1mean < lowlim or Ch2mean < lowlim:
                    check_array[n,i,j] = NaN

        #Actually calculates anisotropy values
        nCH1 = CH1[n,:,:]
        nCH2 = CH2[n,:,:]

        #Different LASERs:
        if FV3000 == 1:
            nCH_pp = (nCH1)
            nCH_pl = (nCH2)
        if FV3000 == 0:
            nCH_pp = (nCH2)
            nCH_pl = (nCH1)

        #Calculates anisotrpy values for all frames
        nGP = process(nCH_pp,nCH_pl,GPFactor)
        nGP = ndimage.median_filter(nGP[:-1,:-1], size = 3)
        nmeanGP = (nanmean(np.multiply(nGP,check_array[n,:,:])))
        meanGPvalues.append(nmeanGP)


        #Inserts each anisotropy element into the 3D array alliso.
        alliso[n,:,:] = np.multiply(nGP,check_array[n,:,:])

        #Specific list:
        specific_list.append(nGP[specific_x,specific_y])

    if delta_r == 1:
        alliso = np.subtract(free_aniso,alliso)


    #The last few frames of each acquisition usually is a bit strange and has large or small values so we remove them.
    alliso = alliso[:-fixnum-1,:,:]




    if anisotropy_pseudocolour == 1:
        print("Creating average pseudocolour map")
        GP_pseudo = np.zeros((CH1.shape[1]-1,CH1.shape[2]-1))
        GP_pseudo[:] = np.nan
        GP_2 = np.zeros((CH1.shape[1]-1,CH1.shape[2]-1))
        GP_2[:] = np.nan
        count = 0
        total = 0
        lowpixels = 0
        highpixels  = 0

        for i in range(CH1.shape[1]-1):
            for j in range(CH1.shape[2]-1):
                if math.isnan(np.average(check_array[:,i,j])) == False:
                    GP_2[i,j] = np.mean(avgfix(alliso[:,i,j],fixnum))

                    if GP_2[i,j] > iso_threshold:
                        GP_pseudo[i,j] = 1
                        lowpixels += 1
                    else:
                        GP_pseudo[i,j] = 0
                        highpixels += 1

        percentage = 1-percentage2d(GP_pseudo,0.5)



    if gif == 1:

        print("Preparing .gif frames.")
        alliso_threshold = np.ones((len(alliso[0]),len(alliso[1]),len(alliso[2])))
        for i in range(len(alliso[0])):
            for j in range(len(alliso[1])):
                for k in range(len(alliso[2])):

                    if delta_r == 1:
                        if alliso[i,j,k] > iso_threshold:
                            alliso_threshold[i,j,k] = 0
                    if delta_r == 0:
                        if alliso[i,j,k] < iso_threshold:
                            alliso_threshold[i,j,k] = 0
                    if math.isnan(check_array[i,j,k]) == True:
                        alliso_threshold[i,j,k] = NaN

        images_iso = []
        images_iso_threshold = []
        for n in range(len(alliso[0])):
            images_iso.append(ndimage.median_filter(alliso[n,:,:], size = 3))
            images_iso_threshold.append(ndimage.median_filter(alliso_threshold[n,:,:], size = 3))



    if standard == 1:
        stddev = []
        isoavg = []
        intensity_list = []

        count = 0

        pseudocolour = np.zeros((CH1.shape[1]-1,CH1.shape[1]-1))
        pseudocolour[:] = NaN

        stdevmap = np.zeros((CH1.shape[1]-1,CH1.shape[1]-1))
        stdevmap[:] = NaN


        a = 100
        per = 0
        total_check = 0


        for i in range(CH1.shape[1]-1):

            per = np.ceil(i / (CH1.shape[1]-1) * 10)
            if a != per:
                print("Standard Deviation:",per*10,"%" )
            a = per

            for j in range(CH1.shape[2]-1):

                if math.isnan(check_array[:,i,j].mean()) == False:

                    list = alliso[:,i,j][:]
                    list = avgfix(list,fixnum)
                    length = len(list)

                    total_check += 1

                    deviation = stats.stdev(list)

                    if deviation != 0:
                        stdevmap[i,j] = deviation
                        stddev.append(deviation)
                        isoavg.append(np.average(list))
                        intensity_list.append((np.mean(CH1[:,i,j])+np.mean(CH2[:,i,j]))/2)

                    if stddev_pseudocolour == 1 and math.isnan(deviation) == False:

                        pseudocolour[i,j] = math.floor((deviation/stdev_interval)) * stdev_interval

                        if deviation < stdev_threshold:
                            count += 1



    if aniso_trace_intensity == 1:
        N = 10
        intensity_list_sorted = sorted(intensity_list)
        median_intensity = intensity_list_sorted[int(len(intensity_list_sorted) / 2)]
        #print(intensity_list_sorted)
        #print(len(intensity_list_sorted))
        lowerintensity = intensity_list_sorted[N]
        upperintensity = intensity_list_sorted[-1 * (N+1)]
        max_intensity = max(maximum2d(CH1[0,:,:])[0],maximum2d(CH2[0,:,:])[0])

        #List of upper and lower intensity standard deviation values.
        upperstdev = []
        upperlist = []
        lowerstdev = []
        lowerlist = []

        upperavg = np.zeros(length)
        loweravg = np.zeros(length)

        print("Plotting by intensity...")

        for i in range(CH1.shape[1]-1):
            for j in range(CH1.shape[2]-1):
                if math.isnan(check_array[:,i,j].mean()) == False:

                    list = alliso[:,i,j][:]
                    list = avgfix(list,fixnum)


                    if (np.mean(CH1[:,i,j])+np.mean(CH2[:,i,j]))/2 > upperintensity:
                        plt.plot(np.arange(len(list)),list, color='orangered', alpha=0.3)

                        upperavg = np.add(upperavg,list)
                        upperlist.append(list)


                        # exporttocsv(list,["High Intensity:"])


                    elif (np.mean(CH1[:,i,j])+np.mean(CH2[:,i,j]))/2 < lowerintensity:
                        plt.plot(np.arange(len(list)),list, color='lime', alpha=0.3)

                        loweravg = np.add(loweravg,list)
                        lowerlist.append(list)


                        # exporttocsv(list,["Low Intensity:"])



                    if (np.mean(CH1[:,i,j])+np.mean(CH2[:,i,j]))/2 > median_intensity:
                        upperstdev.append(stats.stdev(list))

                    elif (np.mean(CH1[:,i,j])+np.mean(CH2[:,i,j]))/2 < median_intensity:
                        lowerstdev.append(stats.stdev(list))

                    # else:
                    #     plt.plot(np.arange(len(list)),list, color='yellow', alpha=0.1)

        file = open('homoFRET_export_5frametraces_HYPOTONIC.csv', 'a+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows([[str(fileCh1)],["High intensity traces:"]])

        for i in range(len(upperlist)):

            file = open('homoFRET_export_5frametraces_HYPOTONIC.csv', 'a+', newline ='')
            with file:
                write = csv.writer(file)
                write.writerows([upperlist[i]])


        file = open('homoFRET_export_5frametraces_HYPOTONIC.csv', 'a+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows([[str(fileCh1)],["Low intensity traces:"]])

        for i in range(len(lowerlist)):

            file = open('homoFRET_export_5frametraces_HYPOTONIC.csv', 'a+', newline ='')
            with file:
                write = csv.writer(file)
                write.writerows([lowerlist[i]])

        upperavg = np.divide(upperavg,N)
        loweravg = np.divide(loweravg,N)


        plt.plot(np.arange(len(upperavg)),upperavg, color='darkred', linewidth=2, label="High Intensity")
        plt.plot(np.arange(len(loweravg)),loweravg, color='green', linewidth=2, label="Low Intensity")
        plt.legend(loc="lower left")
        plt.title("Anisotropy Traces Of High/Low Intensity")
        plt.show()

        plt.hist(upperstdev, np.linspace(0,0.03,100), color='darkred', label="High Intensity", alpha=0.8)
        plt.hist(lowerstdev, np.linspace(0,0.03,100), color='green', label="Low Intensity", alpha=0.8)
        plt.legend(loc="upper left")
        plt.title("StdDev Distribution of High/Low Intensty")
        plt.show()

        plt.hist(intensity_list, np.linspace(0, max_intensity + 5, 100), color='green')
        plt.title("Intensity Distribution")
        plt.show()






    if correlate == 1:

        zero_away = []
        one_away = []
        one_temp = []
        two_away = []
        two_temp = []
        three_away = []
        three_temp = []

        # for i in range(CH1.shape[1]-4):
        #     print("Correlating at row",i)
        #     for j in range(CH1.shape[2]-4):
        #         if math.isnan(check_array[:,i,j].mean()) == False:

        if quick_correlate == 1:
            print("Correlating random locations")
            for k in range(100):
                i = random.randint(3,120)
                j = random.randint(3,120)


                if math.isnan(check_array[:,i,j].mean()) == False:

                    list = alliso[:,i,j][:]
                    list = avgfix(list,fixnum)
                    zero_away.append(scipy.signal.correlate(list,list))
                    one_temp = []
                    two_temp = []
                    three_temp = []

                    #N == 1
                    for n in range(-1,2):
                        for m in range(-1,2):
                            if abs(n) == 1 or abs(m) == 1:

                                if math.isnan(check_array[:,i+n,j+m].mean()) == False:
                                    one = alliso[:,i+n,j+m]
                                    one = avgfix(one,fixnum)
                                    correlation = scipy.signal.correlate(list,one)
                                    one_temp.append(correlation)

                    arrays = [np.array(x) for x in one_temp]
                    one_away.append([np.mean(k) for k in zip(*arrays)])


                    #N == 2
                    for n in range(-2,3):
                        for m in range(-2,3):
                            if abs(n) == 2 or abs(m) == 2:

                                if math.isnan(check_array[:,i+n,j+m].mean()) == False:
                                    two = alliso[:,i+n,j+m]
                                    two = avgfix(two,fixnum)
                                    correlation = scipy.signal.correlate(list,two)
                                    two_temp.append(correlation)

                    arrays = [np.array(x) for x in two_temp]
                    two_away.append([np.mean(k) for k in zip(*arrays)])

                    #N == 2
                    for n in range(-3,4):
                        for m in range(-3,4):
                            if abs(n) == 3 or abs(m) == 3:

                                if math.isnan(check_array[:,i+n,j+m].mean()) == False:
                                    three = alliso[:,i+n,j+m]
                                    three = avgfix(three,fixnum)
                                    correlation = scipy.signal.correlate(list,three)
                                    three_temp.append(correlation)

                    arrays = [np.array(x) for x in three_temp]
                    three_away.append([np.mean(k) for k in zip(*arrays)])



        arrays = [np.array(x) for x in zero_away]
        correlation_0 = [np.mean(k) for k in zip(*arrays)]

        arrays = [np.array(x) for x in one_away]
        correlation_1 = [np.mean(k) for k in zip(*arrays)]

        arrays = [np.array(x) for x in two_away]
        correlation_2 = [np.mean(k) for k in zip(*arrays)]

        arrays = [np.array(x) for x in three_away]
        correlation_3 = [np.mean(k) for k in zip(*arrays)]

        plt.plot(np.arange(len(correlation_0)),correlation_0, label = 'Zero', color='k')
        plt.plot(np.arange(len(correlation_0)),correlation_1, label = 'One', color='r')
        plt.plot(np.arange(len(correlation_0)),correlation_2, label = 'Two', color='y')
        plt.plot(np.arange(len(correlation_0)),correlation_3, label = 'Three', color='g')
        plt.legend()
        plt.show()


    imagelim(CH1[0,:,:],"Intensity Ch1","gist_gray",0,max_intensity)
    imagelim(CH2[0,:,:],"Intensity Ch2","gist_gray",0,max_intensity)

    percentage = 1 - percentage2d(GP_pseudo,0.5)
    print(round(percentage*100,4),"% compact chromatin by average anisotropy threshold pseudoclolour.")

    if delta_r == 1:
        print("Average mean delta r:",nanmean(GP_2))
        print(lowpixels,highpixels,"low, high")
        imagelim(GP_2,"Delta r average","jet",0.05,0.2)
        imagelim(ndimage.median_filter(GP_pseudo, size = 3),"Delta r average pseudocolour","coolwarm",0,1)

    if delta_r == 0:
        print("Average mean anisotropy:",round(nanmean(GP_2),4))
        print(lowpixels,highpixels,"low, high")
        imagelim(GP_2,"Anisotropy average","jet_r",0.2,0.4)
        imagelim(ndimage.median_filter(GP_pseudo, size = 3),"Anisotropy average pseudocolour","coolwarm_r",0,1)



    if standard == 1:
        imagelim(np.multiply(stdevmap,check_array[0,:,:]),"stdevmap","coolwarm_r",stdev_threshold,stdev_threshold + stdev_interval*stdev_interval_num)


        if stddev_pseudocolour == 1:
            pseudocolour = ndimage.median_filter(pseudocolour, size = 3)
            percentage = percentage2d(pseudocolour,lowlimstdev)
            #print(percentage,"% compact chromatin by Stdev. with filter")
            imagelim(np.multiply(pseudocolour,check_array[0,:,:]),"StDev. Pseudocolour","coolwarm_r",stdev_threshold,stdev_threshold + stdev_interval*stdev_interval_num)
            plt.close()

        if export == 1:
            file = open('homoFRET_export.csv', 'a+', newline ='')
            with file:
                write = csv.writer(file)
                write.writerows([[fileCh1[:-8]]])
            exporttocsv(stddev,["Standard Deviation"])
            exporttocsv(isoavg,["Delta r Average (Per pixel)"])



    if gif == 1:
        block = amp.blocks.Imshow(images_iso_threshold)
        anim = amp.Animation([block])

        anim.controls()
        if delta_r == 1:
            anim.save_gif('delta_r_GIF_threshold')
        if delta_r == 0:
            anim.save_gif('anisotropy_GIF_threshold')
        plt.show()


    if power == 1:

        stdevsorted = sorted(stddev)
        anisotropy_list_sorted = sorted(isoavg)
        freqs = np.fft.fftfreq(list.size, frame_time)
        idx = np.argsort(freqs)


        largest_stddev = stdevsorted[-150]
        smallest_stddev = stdevsorted[150]

        largest_iso = anisotropy_list_sorted[-150]
        smallest_iso = anisotropy_list_sorted[150]
        # print("Smallest and largest 100 iso are",smallest_iso,largest_iso)

        if highpower == 1:
            total = 0
            powersp = np.zeros(512)
            for x in range(alliso.shape[1]-1):
                for y in range(alliso.shape[2]-1):
                    list = alliso[:,x,y]
                    if math.isnan(np.mean(list))  == False:
                        list = avgfix(list,fixnum)

                        if np.average(list) > largest_iso:
                            total  += 1
                            ps = np.abs(np.fft.fft(list))**2
                            freqs = np.fft.fftfreq(list.size, frame_time)
                            idx = np.argsort(freqs)
                            powersp = np.add(powersp,ps[idx])

                            # plt.plot(freqs[idx],ps[idx])
                            # plt.xlim(0.002,nyquist)
                            # plt.ylim(-1,10)
                            # plt.title("High Anisotropy TEST (Open Chromatin) Power")
                            # plt.show()



            # print("Total count for high =",total)
            # print("Powersp for HIGH is",powersp)
            # print("Freqs for high is",freqs[idx])
            plt.plot(freqs[idx], np.divide(powersp,total))
            plt.xlim(0.002,nyquist)
            plt.ylim(-1,10)
            plt.title("High Anisotropy (Open Chromatin) Power")
            plt.show()

            exporttocsv(list,["Anisotropy/DeltaR for HIGH Anisotropy"])
            exporttocsv(freqs[idx],["freqs"])
            exporttocsv(np.divide(powersp,total),["ps"])

        if lowpower == 1:
            total = 0
            powersp = np.zeros(512)

            for x in range(alliso.shape[1]-1):
                for y in range(alliso.shape[2]-1):
                    list = alliso[:,x,y]
                    if math.isnan(np.mean(list))  == False:
                        list = avgfix(list,fixnum)

                        if np.average(list) < smallest_iso:
                            total  += 1


                            ps = np.abs(np.fft.fft(list))**2
                            freqs = np.fft.fftfreq(list.size, frame_time)
                            idx = np.argsort(freqs)

                            powersp = np.add(powersp,ps[idx])

            # print("Total count for low =",total)
            # print("Powersp for low is",powersp)
            # print("Freqs for low is",freqs[idx])
            plt.plot(freqs[idx], np.divide(powersp,total))
            plt.xlim(0.002,nyquist)
            plt.ylim(-1,10)
            plt.title("Low Anisotropy (Compact Chromatin) Power")
            plt.show()


            exporttocsv(list,["Anisotropy/DeltaR for LOW Anisotropy"])
            exporttocsv(freqs[idx],["freqs"])
            exporttocsv(np.divide(powersp,total),["ps"])



        if power == 1:
            print("Calculating average power spectrum.")
            powersp = np.zeros(512)
            total = 0

            for x in range(alliso.shape[1]-1):
                for y in range(alliso.shape[2]-1):
                    list = alliso[:,x,y]
                    if math.isnan(np.mean(list))  == False:
                        list = avgfix(list,fixnum)
                        total  += 1

                        ps = np.abs(np.fft.fft(list))**2
                        freqs = np.fft.fftfreq(list.size, frame_time)
                        idx = np.argsort(freqs)

                        powersp = np.add(powersp,ps[idx])

            print("Total count for total =",total)
            plt.plot(freqs[idx], np.divide(powersp,total))
            plt.xlim(0.002,nyquist)
            plt.ylim(-1,10)
            plt.title("Total power")
            plt.show()






    if perovertime == 1:
        perlist = []
        print("Calculating percentage over time.")
        for i in range(alliso.shape[0]):
            perlist.append(percentage2d(alliso[i,:,:],iso_threshold))
        perlist = perlist[:-fixnum-1]
        if delta_r == 0:
            perlist = np.subtract(1,perlist)
        plt.plot(np.arange(len(perlist)),perlist)
        if delta_r == 1:
            plt.title("Percentage over time of delta r")
        if delta_r == 0:
            plt.title("Percentage over time of anisotropy")
        plt.show()
        print("Average percentage:",100 * round(np.mean(perlist),4))

        # powersp = np.zeros(256)
        # ps = np.abs(np.fft.fft(perlist))**2
        # freqs = np.fft.fftfreq(list.size, frame_time)
        # idx = np.argsort(freqs)
        # powersp = np.add(powersp,ps[idx])


        ps = np.abs(np.fft.fft(perlist))**2

        freqs = np.fft.fftfreq(perlist.size, frame_time)
        idx = np.argsort(freqs)
        plt.plot(freqs[idx], ps[idx])
        plt.xlim(0.001,nyquist)
        plt.ylim(-1,10)
        plt.title("Percentage over time power spectrum")
        plt.show()

    #Export anisotropy mean and specific to .csv
    if export == 1:
        if delta_r == 1:
            exporttocsv(meanGPvalues,["Mean delta r Value (Per frame)"])
            exporttocsv(specific_list,["Specific delta r Value"])
        if delta_r == 0:
            exporttocsv(meanGPvalues,["Mean anisotropy Value (Per frame)"])
            exporttocsv(specific_list,["Specific anisotropy Value"])
        file = open('homoFRET_export.csv', 'a+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows([[""]])

    if specific == 1:
        plt.plot(np.arange(len(specific_list)),specific_list)
        if delta_r == 1:
            plt.title("Specific ROI delta r")
        if delta_r == 0:
            plt.title("Specific ROI anisotropy")
        plt.show()
        if delta_r == 1:
            print("Average delta r and Standard Deviation at",specific_x,specific_y,":",np.mean(specific_list),stats.stdev(specific_list))
        if delta_r == 0:
            print("Average anisotropy and Standard Deviation at",specific_x,specific_y,":",np.mean(specific_list),stats.stdev(specific_list))
        ps = np.abs(np.fft.fft(specific_list))**2
        freqs = np.fft.fftfreq(specific_list.size, frame_time)
        idx = np.argsort(freqs)
        plt.plot(freqs[idx], ps[idx])
        plt.xlim(0.001,nyquist)
        plt.ylim(-1,10)
        plt.title("Power spectrum of specific ROI.")
        plt.show()
