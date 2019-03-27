import os
from math import ceil
import numpy as np
from emily import *


def finalise_gen(path,final_size=10000):

    files=dirlist(path)
    merging_flag=False

    print('Running diagnostics for file: '+path+'  ... \n') 
    py, dat=[],[]
    for file in files:
        if file[-3:]=='.py':
            py.append(file)
        elif file[-4:]=='.mat':  #In the future the type of file might change
            dat.append(file)

    # Checking if all files that should have been generated from auto_gen are her
    if len(dat)==len(py):
        print('Files succesfully generated, all files are here')
        print(len(dat),' out of ',len(py))
        merging_flag=True  # Declaring that merging can happen now
    
    elif len(dat)>len(py):
        print('There are some files already merged or finalised\n')
        print('Files succesfully generated, all files are here')
        print(len(dat),' out of ',len(py))
        merging_flag=True  # Declaring that merging can happen now

    
    else:
        failed_py=[]
        for i in range(len(py)):
            py_id=py[i][4:-3]
            counter=0
            for dataset in dat:
                if py_id in dataset:
                    counter=1
            if counter==0:
                print(py[i],' failed to proceed')
                failed_py.append(py[i])



    if merging_flag==False:
        print('\n\nDo you want to try and run again the failed procedurs? y/n')
        answer1=input()
        
        if answer1 in ['Y','y','yes']:
            with open(path+'/auto_gen_redo.sh','w') as fnew:
                fnew.write('#!/bin/sh\n\n')
                with open(path+'/auto_gen.sh','r') as f:
                    for line in f:
                        for pyid in failed_py:
                            if pyid in line:
                                fnew.write(line+'\n')
                                break

            print('All set. The following generators are going to run again:\n')
            for pyid in failed_py:
                print(pyid)

            print('\n\nInitiate dataset generation y/n?')
            answer2=input()

            if answer2 in ['y','Y','yes']:
                os.system('sh '+path+'/auto_gen_redo.sh')
                return
            else:
                print('Data generation canceled')
                os.system('rm '+path+'/auto_gen_redo.sh')
                return
            
            
        elif answer1 in ['N','n','no','exit']:
            print('Exiting procedure')
            return
        
        
    if merging_flag==True:
        
        print('\n Do you wanna proceed to the merging of the datasets? y/n')
        answer3=input()
              
        if answer3 in ['n','N','no','NO','No']:
            print('Exiting procedure')
            return
        
        elif answer3 in ['y','Y','yes','Yes']:        
        
            lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  # Labels used in saving file
            if final_size not in lab:
                lab[final_size]=str(final_size)
            
            # Geting the details of files from their names
            set_name,set_id, set_size,ids,new_dat=[],[],[],[],[]
            for dataset in dat:
                if 'part_of' not in dataset:
                    new=dataset.split('_')
                    
                    if 'SNR' in dataset:
                        new.pop(-1)
                        new.pop(-1)

                        final_name=('_'.join(new)+'_'+lab[final_size]+'.mat')
                    elif 'noise' in dataset.split('_')[-3]:
                        new.pop(-1)
                        new[-1]='No'+new[-1]
                        final_name=('_'.join(new)+'_'+lab[final_size]+'.mat')

                    os.system('mv '+path+dataset+' '+path+final_name)

                else:
                    new_dat.append(dataset)
                    new=dataset.split('part_of_')
                    set_name.append(dataset)
                    set_id.append(new[1].split('_')[0])
                    set_size.append(new[1].split('_')[1].split('.')[0])
                    if new[1].split('_')[0] not in ids: ids.append(new[1].split('_')[0])
                    
                    
            dat=new_dat

            # Creating the inputs for the function data_fusion
            merging_index=[]
            for ID in ids:
                sets,sizes=[],[]
                for i in range(len(dat)):
                    
                    if set_id[i]==ID:
                        sets.append(dat[i])
                        sizes.append(set_size[i])
                merging_index.append([ID,sets,sizes])


            # Initiate merging of files
            for k in range(len(merging_index)):
                if 'SNR' in merging_index[k][1][0].split('part_of_')[0][-6:]:
                    final_name=merging_index[k][1][0].split('part_of_')[0]+lab[final_size]+'.mat'
                if 'noise' in merging_index[k][1][0].split('part_of_')[0][-6:]:
                    final_name=merging_index[k][1][0].split('part_of_')[0]+'No'+ids[k]+'_'+lab[final_size]+'.mat'



                data_fusion(names=merging_index[k][1]
                    ,sizes=None # Given the sum is right
                    ,save=path+final_name
                    ,data_source_file=path)    

            print('\n\nDataset is complete!')
             
                
            # Deleting unnescesary file in the folder
            print('\n Do you want to delete unnecessary files? y/n')
            
            answer4=input()
            if answer4 in ['n','N','no','NO','No']:
                return

            elif answer4 in ['y','Y','yes','Yes']: 

                for file in dirlist(path):
                    
                    if ('info' not in file) and (lab[final_size] not in file):

                        os.system('rm '+path+file)
                
                
                print('File is finalised')
                return



    



            
    
                    
        
        

        












    



            
    
                    
        
        













            
    
                    
        
        









