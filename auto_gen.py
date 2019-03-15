import os
from math import ceil
import numpy as np

# This function generates automaticaly training sets given the options. After it is run you need to run the next one to give the dataset its final form

def auto_gen(set_type
             ,date_ini
             ,size=10000
             ,fs=2048
             ,length=4 
             ,lags=11
             ,t=32
             ,s_name=''):
    
    def dirlist(filename):           # Creates a list of names of the files                   
        fn=os.listdir(filename)      # we are interested to use so that we can call 
        for i in range(0,len(fn)-1): # them automaticaly all them automaticaly
            if fn[i][0]=='.':
                fn.pop(i)
        fn.sort()
        return fn
    # 'set_type' must have one of the following formats:
    # noise --> ['noise', 'real'/'sudo_real', detectors(str) , number of sets (int)],(optinoal:) spectogram(b), phasegram(b), res(int)]
    # cbc --> ['cbc','real'/'sudo_real', detectors(str),'cbc_02',[20, 30, 40] ,(optinoal:) spectogram(b), phasegram(b), res(int)]
    
    
    if len(set_type) not in [4,5,6,7,8]:
        raise TypeError('\'set_type\' must have one of the following formats: \n'+
                        'noise --> [\'noise\', \'real\',detectors, number of sets (int)] \n'+
                        'cbc --> [\'cbc\',\'real\', detectors ,\'cbc_02\',[20, 30, 40 , ...]]')
        
    if set_type[0]=='noise':
        
        
        if set_type[1] in ['sudo_real','real']: 
            noise_type=set_type[1]
        else:
            raise ValueError('noise can be either real or sudo_real')
            
        # detectors
        for d in set_type[2]:
            if (d!='H' and d!='L' and d!='V' and d!='K'): 
                raise ValueError('Not acceptable character for detector. You should include: \nH for LIGO Hanford\nL for LIGO Livingston \nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')
        
        if (isinstance(set_type[3],int) and set_type[3]>0):
            num_of_sets=set_type[3] 
        else:
            raise ValueError('Number of sets must be integer positive')
            
        spect_b=False
        phase_b=False
        res=128
        if len(set_type)==7:
            spect_b=set_type[4]
            phase_b=set_type[5]
            res=set_type[6]

            
    if set_type[0]=='cbc':
        
        if set_type[1] in ['sudo_real','real']: 
            noise_type=set_type[1]
        else:
            raise ValueError('noise can be either real or sudo_real')
            
        # detectors
        for d in set_type[2]:
            if (d!='H' and d!='L' and d!='V' and d!='K'): 
                raise ValueError('Not acceptable character for detector. You should include: \nH for LIGO Hanford\nL for LIGO Livingston \nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')


        if set_type[3] in dirlist('/home/vasileios.skliris/EMILY/injections/cbcs'):
            injection_source=set_type[3]
        else:
            raise ValueError('File does not exist')
            
        if len(set_type[4]) > 0:
            for num in set_type[4]:
                if not ((isinstance(num, float) or isinstance(num,int)) and num >=0 ):
                    raise ValueError('SNR values have to be possitive numbers')
                else:
                    snr_list=set_type[4]
                    num_of_sets=len(snr_list)
        else:
            raise TypeError('SNR must be specified as a list [20, 15 , ]')
         
        spect_b=False
        phase_b=False
        res=128
        if len(set_type)==8:
            spect_b=set_type[5]
            phase_b=set_type[6]
            res=set_type[8]
    # Calculation of how many segments we need for the given requirements. There is a starting date and if it needs more it goes to the next one.

    
    date_list=dirlist('/home/vasileios.skliris/EMILY/ligo_data/2048')

    date=date_list[date_list.index(date_ini)]     # Generating a list with all the available dates
    counter=date_list.index(date_ini)             # Calculating the index of the initial date

    
    # Calculation of the duration 
    # Here we infere the duration needed given the lags used in the method

    if lags==1:
        duration_need = size*num_of_sets*length
        tail_crop=0
    if lags%2 == 0:
        duration_need = ceil(size*num_of_sets/(lags*(lags-2)))*lags*length
        tail_crop=lags*(lags-2)
    if lags%2 != 0 and lags !=1 :
        duration_need = ceil(size*num_of_sets/(lags*(lags-1)))*lags*length
        tail_crop=lags*(lags-1)


    # Creation of lists that indicate characteristics of the segments based on the duration needed. 
    # These are used for the next step.

    duration_total = 0
    duration, gps_time, seg_list=[],[],[]

    while duration_need > duration_total:
        counter+=1
        segments=dirlist('/home/vasileios.skliris/EMILY/ligo_data/'+str(fs)+'/'+date+'/H')
        print(date)
        for seg in segments:
            for j in range(len(seg)):
                if seg[j]=='_': 
                    gps, dur = seg[j+1:-5].split('_')
                    break

            duration.append(int(dur))
            gps_time.append(int(gps))
            seg_list.append([date,seg])

            duration_total+=(int(dur)-2*t-tail_crop)   # I initialy had 2 insted of 3 but it was overflowing
            print('    '+seg)

            if duration_total > duration_need: break

        if counter==len(date_list): counter==0        
        date=date_list[counter]



    #for i in range(len(np.array(seg_list)[:,1])):
    #    print(np.array(seg_list)[:,0][i],np.array(seg_list)[:,1][i],duration[i],gps_time[i])



    # Generation of lists with the parameters that the dataset_generator_noice will use to make 
    # the automated datasets.

    size_list=[]            # List with the sizes for each generation of noise 
    starting_point_list=[]  # List with the starting points for each generation of noise (seconds)
    seg_list_2=[]           # List with the segment names for each generation of noise
    number_of_set=[]        # List with the number of set that this generation of noise will go
    name_list=[]            # List with the name of the set to be generated
    number_of_set_counter=0 # Counter that keeps record of how many instantiations have left to be generated to complete a set
    
    
    if set_type[0]=='noise':
        set_num=1

        for i in range(len(np.array(seg_list)[:,1])):

            # local size indicates the size of the file left for generation of datasets, 
            # when it is depleted the algorithm moves to the next segment.
            # Here we infere the local size given the lags used in the method

            if lags==1:    # zero lag case
                local_size=ceil((duration[i]-2*t-tail_crop)/length)
            if lags%2 == 0:
                local_size=ceil((duration[i]-2*t-tail_crop)/length/lags)*lags*(lags-2)
            if lags%2 != 0 and lags !=1 :
                local_size=ceil((duration[i]-2*t-tail_crop)/length/lags)*lags*(lags-1)

            # starting point always begins with the window of the psd to avoid deformed data of the begining    
            local_starting_point=t

            # There are three cases when a segment is used.
            # 1. That it has to fill a previous set first and then move to the next
            # 2. That it is the first set so there is no previous set to fill
            # 3. It is the too small to fill so its only part of a set.
            # Some of them go through all the stages

            if (len(size_list) > 0 and number_of_set_counter > 0 and local_size >= size-number_of_set_counter):

                size_list.append(size-number_of_set_counter)          # Saving the size of the generation
                seg_list_2.append(seg_list[i])                        # Saving the name of the date file and seg used
                starting_point_list.append(local_starting_point)      # Savint the startint_point of the generation

                #Update the the values for the next set
                local_size-=(size-number_of_set_counter)
                if lags==1: local_starting_point+=(size-number_of_set_counter)*length
                if lags%2 == 0: local_starting_point+=(ceil((size-number_of_set_counter)/lags/(lags-2))*lags*length)
                if lags%2 != 0 and lags !=1 : local_starting_point+=(ceil((size-number_of_set_counter)/lags/(lags-1))*lags*length)
                number_of_set_counter+=(size-number_of_set_counter)

                # If this generation completes the size of a whole set (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(set_num)
                    if size_list[-1]==size: 
                        name_list.append('No_'+str(set_num)+'_')
                    else:
                        name_list.append('part_of_'+str(set_num))
                    set_num+=1
                    number_of_set_counter=0
                    if set_num > num_of_sets: break

                elif number_of_set_counter < size:
                    number_of_set.append(set_num)
                    name_list.append('part_of_'+str(set_num))


            if (len(size_list) == 0 or number_of_set_counter==0):
                while local_size >= size:
                    # Generate data with size 10000 with final name of 'name_counter'
                    size_list.append(size)
                    seg_list_2.append(seg_list[i])
                    starting_point_list.append(local_starting_point)

                    #Update the the values for the next set
                    local_size-=size
                    if lags==1: local_starting_point+=size*length
                    if lags%2 == 0: local_starting_point+=(ceil(size/lags/(lags-2))*lags*length)
                    if lags%2 != 0 and lags !=1 : local_starting_point+=(ceil(size/lags/(lags-1))*lags*length)
                    number_of_set_counter+=size

                    # If this generation completes the size of a whole set (with size=size) it changes the labels
                    if number_of_set_counter == size:
                        number_of_set.append(set_num)
                        if size_list[-1]==size: 
                            name_list.append('No_'+str(set_num)+'_')
                        else:
                            name_list.append('part_of_'+str(set_num))
                        set_num+=1
                        if set_num > num_of_sets: break
                        number_of_set_counter=0


            if (local_size < size and local_size >0):
                # Generate data with size 'local_size' with local name to be fused with later one
                size_list.append(local_size)
                seg_list_2.append(seg_list[i])
                starting_point_list.append(local_starting_point)

                #Update the the values for the next set
                number_of_set_counter+=local_size  # Saving a value for what is left for the next seg to generate

                # If this generation completes the size of a whole set (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(set_num)
                    if size_list[-1]==size: 
                        name_list.append('No_'+str(set_num)+'_')
                    else:
                        name_list.append('part_of_'+str(set_num))
                    set_num+=1
                    if set_num > num_of_sets: break
                    number_of_set_counter=0

                elif number_of_set_counter < size:
                    number_of_set.append(set_num)
                    name_list.append('part_of_'+str(set_num))





    
    if set_type[0]=='cbc':
        
        set_num=0

        for i in range(len(np.array(seg_list)[:,1])):
                

            # local size indicates the size of the file left for generation of datasets, 
            # when it is depleted the algorithm moves to the next segment.
            # Here we infere the local size given the lags used in the method

            if lags==1:    # zero lag case
                local_size=ceil((duration[i]-2*t-tail_crop)/length)
            if lags%2 == 0:
                local_size=ceil((duration[i]-2*t-tail_crop)/length/lags)*lags*(lags-2)
            if lags%2 != 0 and lags !=1 :
                local_size=ceil((duration[i]-2*t-tail_crop)/length/lags)*lags*(lags-1)
                

            # starting point always begins with the window of the psd to avoid deformed data of the begining    
            local_starting_point=t

            # There are three cases when a segment is used.
            # 1. That it has to fill a previous set first and then move to the next
            # 2. That it is the first set so there is no previous set to fill
            # 3. It is the too small to fill so its only part of a set.
            # Some of them go through all the stages

            if (len(size_list) > 0 and number_of_set_counter > 0 and local_size >= size-number_of_set_counter):

                size_list.append(size-number_of_set_counter)          # Saving the size of the generation
                seg_list_2.append(seg_list[i])                        # Saving the name of the date file and seg used
                starting_point_list.append(local_starting_point)      # Savint the startint_point of the generation

                #Update the the values for the next set
                local_size-=(size-number_of_set_counter)                     
                if lags==1: local_starting_point+=(size-number_of_set_counter)*length
                if lags%2 == 0: local_starting_point+=(ceil((size-number_of_set_counter)/lags/(lags-2))*lags*length)
                if lags%2 != 0 and lags !=1 : local_starting_point+=(ceil((size-number_of_set_counter)/lags/(lags-1))*lags*length)
                number_of_set_counter += (size-number_of_set_counter)

                # If this generation completes the size of a whole set (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(snr_list[set_num])
                    if size_list[-1]==size: 
                        name_list.append(str(snr_list[set_num])+s_name)
                    else:
                        name_list.append('part_of_'+str(snr_list[set_num])+s_name)
                    set_num+=1
                    number_of_set_counter=0
                    if set_num >= num_of_sets: break
                        
                elif number_of_set_counter < size:
                    number_of_set.append(snr_list[set_num])
                    name_list.append('part_of_'+str(snr_list[set_num])+s_name)

            if (len(size_list) == 0 or number_of_set_counter==0):
                while local_size >= size:
                    # Generate data with size 10000 with final name of 'name_counter'
                    size_list.append(size)
                    seg_list_2.append(seg_list[i])
                    starting_point_list.append(local_starting_point)

                    #Update the the values for the next set
                    local_size-=size
                    if lags==1: local_starting_point+=size*length
                    if lags%2 == 0: local_starting_point+=(ceil(size/lags/(lags-2))*lags*length)
                    if lags%2 != 0 and lags !=1 : local_starting_point+=(ceil(size/lags/(lags-1))*lags*length)
                    number_of_set_counter+=size

                    # If this generation completes the size of a whole set (with size=size) it changes the labels
                    if number_of_set_counter == size:
                        number_of_set.append(snr_list[set_num])
                        if size_list[-1]==size: 
                            name_list.append(str(snr_list[set_num])+s_name)
                        else:
                            name_list.append('part_of_'+str(snr_list[set_num])+s_name)
                        set_num+=1
                        if set_num >= num_of_sets: break
                        number_of_set_counter=0


            if (local_size < size and local_size >0):
                # Generate data with size 'local_size' with local name to be fused with later one
                size_list.append(local_size)
                seg_list_2.append(seg_list[i])
                starting_point_list.append(local_starting_point)

                #Update the the values for the next set
                number_of_set_counter+=local_size  # Saving a value for what is left for the next seg to generate

                # If this generation completes the size of a whole set (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(snr_list[set_num])
                    if size_list[-1]==size: 
                        name_list.append(str(snr_list[set_num])+s_name)
                    else:
                        name_list.append('part_of_'+str(snr_list[set_num])+s_name)
                    set_num+=1
                    if set_num >= num_of_sets: break
                    number_of_set_counter=0

                elif number_of_set_counter < size:
                    number_of_set.append(snr_list[set_num])
                    name_list.append('part_of_'+str(snr_list[set_num])+s_name)
                

    d={'segment' : seg_list_2, 'size' : size_list , 'start_point' : starting_point_list, 'set' : number_of_set, 'name' : name_list}

    print('These are the details of the datasets to be generated: \n')
    for i in range(len(d['segment'])):
        print(d['segment'][i], d['size'][i], d['start_point'][i] ,d['name'][i])
        
    print('Should we proceed to the generation of the following data y/n ? \n \n')

    answer=input()
    
    if answer in ['no','n']:
        print('Exiting procedure')
        return
        
    elif answer in ['yes','y']:
        
        print('Of course lord commander ...')
        print('Type the name of the dataset directory:')
        
        dir_name=input()
        
        if set_type[0]=='noise':
            path='/home/vasileios.skliris/EMILY/datasets/noise/'+set_type[1]+'/'
        elif set_type[0]=='cbc':
            path='/home/vasileios.skliris/EMILY/datasets/'+set_type[0]+'/'

        print('The current path of the directory is: \n'+path+'\n')
        print('Do you wanna change the path y/n ?')
        
        answer2=input()

        if answer2 in ['yes','y']:
            print('Insert complete new path ex. /home/....')
            path=input()
            if path=='exit': return
            
            while not (os.path.isdir(path)):
                print('Path: '+path+' is not existing. Try again or type exit to exit the procedure.')
                path=input()
                if path=='exit': return
            
        if answer2 in ['no','n']:
            if os.path.isdir(path+dir_name):
                print('Already existing '+dir_name+' directory, do want to delete it? y/n')
                answer3=input()
                if answer3=='y':
                    os.system('rm -r '+path+dir_name)
                elif answer3=='n':
                    return
            print('Initiating procedure ...')
            os.system('mkdir '+path+dir_name)
            print('Creation of directory complete: '+path+dir_name)
            
        os.system('cd '+path+dir_name)
                

        for i in range(len(d['segment'])):

            with open(path+dir_name+'/'+'gen_'+d['name'][i]+'_'+str(d['size'][i])+'.py','w') as f:
                f.write('import sys \n')
                f.write('sys.path.append(\'/home/vasileios.skliris/EMILY\')\n')
                f.write('from emily import * \n')
                
                if set_type[0]=='cbc':

                    comand=('data_generator_cbc(parameters=[\''+injection_source+'\',\''+noise_type+'\','+str(d['set'][i])+']'+
                               ',length='+str(length)+           
                               ',fs='+str(fs)+              
                               ',size='+str(d['size'][i])+             
                               ',detectors=\''+set_type[2]+'\''+
                               ',spec='+str(spect_b)+
                               ',phase='+str(phase_b)+
                               ',res='+str(res)+
                               ',noise_file='+str(d['segment'][i])+
                               ',t='+str(t)+             
                               ',batch_size='+str(lags)+
                               ',starting_point='+str(d['start_point'][i])+
                               ',name=\''+str(d['name'][i])+'_\''+
                               ',destination_path=\''+path+dir_name+'/\''+
                               ',demo=False)')
                    
                elif set_type[0]=='noise':
                    
                    comand=('data_generator_noise(noise_type=\''+noise_type+'\''+
                               ',length='+str(length)+           
                               ',fs='+str(fs)+              
                               ',size='+str(d['size'][i])+             
                               ',detectors=\''+set_type[2]+'\''+
                               ',spec='+str(spect_b)+
                               ',phase='+str(phase_b)+
                               ',res='+str(res)+
                               ',noise_file='+str(d['segment'][i])+
                               ',t='+str(t)+             
                               ',batch_size='+str(lags)+
                               ',starting_point='+str(d['start_point'][i])+
                               ',name=\''+str(d['name'][i])+'\''+
                               ',destination_path=\''+path+dir_name+'/\''+
                               ',demo=False)')
                
                f.write(comand+'\n')

        with open(path+dir_name+'/'+'auto_gen.sh','w') as f2:
            
            f2.write('#!/bin/sh \n\n')
            for i in range(len(d['segment'])):
                
                f2.write('nohup python '+path+dir_name+'/'+'gen_'+d['name'][i]+'_'+str(d['size'][i])+'.py > ' +path+dir_name+'/out_'+d['name'][i]+'_'+str(d['size'][i])+'.out & \n' )
                
        
        with open(path+dir_name+'/info.txt','w') as f3:
            f3.write('INFO ABOUT DATASETS GENERATION \n\n\n\n\n')
            for i in range(len(d['segment'])):
                f3.write(d['segment'][i][0]+' '+d['segment'][i][1]+' '+str(d['size'][i])+' '+str(d['start_point'][i])+'_'+d['name'][i]+'\n')
            
        print('All set. Initiate dataset generation y/n?')
        answer4=input()
        
        if answer4 in ['y','Y']:
            os.system('sh '+path+dir_name+'/auto_gen.sh')
            return
        else:
            print('Data generation canceled')
            os.system('cd')
            os.system('rm -r '+path+dir_name)
            return

                

                
    return
        

        



    
        

        
        
    