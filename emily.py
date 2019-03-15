from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import tight_layout
import matplotlib.pyplot as plt

from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import pickle
from math import ceil

from keras import optimizers, initializers
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

# Keras layers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense,Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, concatenate, Input ,Activation
from keras.layers import Dropout, BatchNormalization

import os
from random import shuffle

from scipy.signal import spectrogram
from simulateddetectornoise import *
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict

################################################################################################################################
################################################################################################################################
################################################################################################################################


def SNR(data,injection,background,maximum=True,fs=2048):
    data=np.array(data)
    temp=np.array(injection)
    back=np.array(background)

    data_fft=np.fft.fft(data)
    template_fft = np.fft.fft(temp)

    # -- Calculate the PSD of the data
    power_data, freq_psd = plt.psd(back, Fs=fs, NFFT=fs, visible=False)

    # -- Interpolate to get the PSD values at the needed frequencies
    datafreq = np.fft.fftfreq(data.size)*fs
    power_vec = np.interp(datafreq, freq_psd, power_data)
    # -- Calculate the matched filter output
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)
    # -- Normalize the matched filter output
    df = np.abs(datafreq[1] - datafreq[0])
    sigmasq = 2*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = abs(optimal_time) / (sigma)
    
    if maximum==True:
        return(max(SNR))
    else:
        return(SNR)
   

    
def dirlist(filename):           # Creates a list of names of the files                   
    fn=os.listdir(filename)      # we are interested to use so that we can call 
    for i in range(0,len(fn)-1): # them automaticaly all them automaticaly
        if fn[i][0]=='.':
            fn.pop(i)
    fn.sort()
    return fn

#def load_noise(name,noise_file,detector): #  Loading the file with the real noise segment.
#    noise=[]                              
#    f_noise = open('/home/vasileios.skliris/EMILY/ligo_data/2048/'+noise_file+'/'+detector+'/'+name,'r') 
#    for line in f_noise:              
#        noise.append(float(line))     
#    f_noise.close()
#    return np.array(noise)
    
def load_inj(dataset,name,detector):    #  Loading the file with the template of the signal,
    inj=[]                      #  which is projected for the detector
    f_inj = open('/home/vasileios.skliris/EMILY/injections/cbcs/'+dataset+'/'+detector+'/'+name,'r') 
    for line in f_inj:              
        inj.append(float(line))     
    f_inj.close()
    return np.array(inj)

def isPrime(n): 
      
    # Corner case 
    if n <= 1 : 
        return False
  
    # check from 2 to n-1 
    for i in range(2, n): 
        if n % i == 0: 
            return False
  
    return True

def load_noise(fs,date_file,detector,name,ind='all'): #  Loading the file with the real noise segment.
    noise=[]
    with open('/home/vasileios.skliris/EMILY/ligo_data/'+str(fs)+'/'+date_file+'/'+detector+'/'+name,'r') as f:
        if isinstance(ind,str) and ind=='all':
            for line in f: noise.append(float(line))
        else:
            for i in range(0,ind[0]):
                next(f)
            for i in ind:
                noise.append(float(next(f)))

    return np.array(noise)

def index_combinations(detectors
                       ,inst
                       ,length
                       ,fs
                       ,size
                       ,start_from_sec=0):
    
    indexes={}

    if inst==1:
        for det in detectors:
            indexes[det]=np.arange(start_from_sec*fs,start_from_sec*fs+size*length*fs, length*fs)
            
    elif inst>=len(detectors):
        
        batches=int(ceil(size/(inst*(inst-1))))
        
        for det in detectors:
            indexes[det]=np.zeros(inst*(inst-1)*batches)

        d=np.int_(range(inst))

        if len(detectors)==1:
            indexes[detectors[0]]=np.arange(start_from_sec*length*fs,(start_from_sec+size)*length*fs, length*fs)



        elif len(detectors)==2:
            for b in range(0,batches):
                for j in range(1,inst):
                    indexes[detectors[0]][(b*(inst-1)+j-1)*inst:(b*(inst-1)+j)*inst]=(start_from_sec+(b*inst+d)*length)*fs
                    indexes[detectors[1]][(b*(inst-1)+j-1)*inst:(b*(inst-1)+j)*inst]=(start_from_sec+(b*inst+np.roll(d,j))*length)*fs
            t1=time.time()

        elif len(detectors)==3:
            for b in range(0,batches):
                for j in range(1,inst):

                    indexes[detectors[0]][(b*(inst-1)+j-1)*inst:(b*(inst-1)+j)*inst]=(start_from_sec+(b*inst+d)*length)*fs
                    indexes[detectors[1]][(b*(inst-1)+j-1)*inst:(b*(inst-1)+j)*inst]=(start_from_sec+(b*inst+np.roll(d,j))*length)*fs
                    indexes[detectors[2]][(b*(inst-1)+j-1)*inst:(b*(inst-1)+j)*inst]=(start_from_sec+(b*inst+np.roll(d,-j))*length)*fs

    for det in detectors:
        indexes[det]=np.int_(indexes[det][:size])

    return(indexes)

##################################################################################################################################################################################################################################################################
    
def conv_model_1D(parameter_matrix, INPUT_SHAPE, LR, verbose=True):
    
    ####################################################
    ##  This type of format for the input data is     ##
    ##  suggested to avoid errors:                    ##
    ##                                                ##
    ##  CORE =     ['C','C','C','C','F','D','D','DR'] ##
    ##  MAX_POOL = [ 2,  0,  0,  2,  0,  0,  0,  0  ] ##
    ##  FILTERS =  [ 8, 16, 32, 64,  0, 64, 32,  0.3] ##
    ##  K_SIZE =   [ 3,  3,  3,  3,  0,  0,  0,  0  ] ## 
    ##                                                ##
    ##  PM=[CORE,MAX_POOL,FILTERS,K_SIZE]             ## 
    ##  in_shape = (52,80,3)                          ##
    ##  lr = 0.00002                                  ##
    ####################################################
    
    CORE = parameter_matrix[0]
    MAX_POOL = parameter_matrix[1]
    FILTERS = parameter_matrix[2]
    K_SIZE = parameter_matrix[3]

    
    model=Sequential()
    receipt=[]

    inis=initializers.he_normal(seed=None)

    for i in range(0,len(CORE)):

        if CORE[i]=='C' and i==0:
            model.add(Conv1D(filters=FILTERS[i]
                             , kernel_size=K_SIZE[i]
                             , activation='elu'
                             , input_shape=INPUT_SHAPE
                             , kernel_initializer=inis,bias_initializer=initializers.Zeros()))
            model.add(BatchNormalization())

            receipt.append('INPUT <------ SHAPE: '+str(INPUT_SHAPE))
            receipt.append('CONV 1D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')

        if CORE[i]=='C' and i!=0:
            model.add(Conv1D(filters=FILTERS[i], kernel_size=K_SIZE[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('CONV 1D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')


        if MAX_POOL[i]!=0 :
            model.add(MaxPool1D(pool_size=MAX_POOL[i], strides=MAX_POOL[i]))
            model.add(BatchNormalization())
            receipt.append('MAX POOL 1D --> KERNEL SHAPE: %d STRIDE: %d ' % (MAX_POOL[i], MAX_POOL[i]))
            receipt.append('Bach Normalization')

        if CORE[i]=='F': 
            model.add(Flatten())
            receipt.append('<---- FLATTEN ---->')

        if CORE[i]=='D':
            model.add(Dense(FILTERS[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('DENSE --> FILTERS: %3d' % FILTERS[i])
            receipt.append('Bach Normalization')


        if CORE[i]=='DR': 
            model.add(Dropout(FILTERS[i]))
            receipt.append('DROP OUT --> '+str(int(100*FILTERS[i]))+'  % ' )

    model.add(Dense(2, activation='softmax'))
    receipt.append('OUTPUT --> SOFTMAX 2 CLASSES')

    opt=optimizers.Nadam(lr=LR
                         , beta_1=0.9
                         , beta_2=0.999
                         , epsilon=1e-8
                         , schedule_decay=0.000002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if verbose==True:
        for line in receipt:
            print(line)

    return(model)

#################################################################################################################################
#################################################################################################################################

def conv_model_2D(parameter_matrix, INPUT_SHAPE, LR, verbose=True):
    
    ####################################################
    ##  This type of format for the input data is     ##
    ##  suggested to avoid errors:                    ##
    ##                                                ##
    ##  CORE =     ['C','C','C','C','F','D','D','DR'] ##
    ##  MAX_POOL = [ 2,  0,  0,  2,  0,  0,  0,  0  ] ##
    ##  FILTERS =  [ 8, 16, 32, 64,  0, 64, 32,  0.3] ##
    ##  K_SIZE =   [ 3,  3,  3,  3,  0,  0,  0,  0  ] ## 
    ##                                                ##
    ##  PM=[CORE,MAX_POOL,FILTERS,K_SIZE]             ## 
    ##  in_shape = (52,80,3)                          ##
    ##  lr = 0.00002                                  ##
    ####################################################
    
    
    CORE = parameter_matrix[0]
    MAX_POOL = parameter_matrix[1]
    FILTERS = parameter_matrix[2]
    K_SIZE = parameter_matrix[3]

    
    model=Sequential()
    receipt=[]

    inis=initializers.he_normal(seed=None)

    for i in range(0,len(CORE)):

        if CORE[i]=='C' and i==0:
            model.add(Conv2D(filters=FILTERS[i]
                             , kernel_size=K_SIZE[i]
                             , activation='elu'
                             , input_shape=INPUT_SHAPE
                             , kernel_initializer=inis,bias_initializer=initializers.Zeros()))
            model.add(BatchNormalization())

            receipt.append('INPUT <------ SHAPE: '+str(INPUT_SHAPE))
            receipt.append('CONV 2D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')

        if CORE[i]=='C' and i!=0:
            model.add(Conv2D(filters=FILTERS[i], kernel_size=K_SIZE[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('CONV 2D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')


        if MAX_POOL[i]!=0 :
            model.add(MaxPool2D(pool_size=(MAX_POOL[i],MAX_POOL[i]), strides=MAX_POOL[i]))
            model.add(BatchNormalization())
            receipt.append('MAX POOL 2D --> KERNEL SHAPE:[%d X %d] STRIDE: %d ' % (MAX_POOL[i], MAX_POOL[i], MAX_POOL[i]))
            receipt.append('Bach Normalization')

        if CORE[i]=='F': 
            model.add(Flatten())
            receipt.append('<---- FLATTEN ---->')

        if CORE[i]=='D':
            model.add(Dense(FILTERS[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('DENSE --> FILTERS: %3d' % FILTERS[i])
            receipt.append('Bach Normalization')


        if CORE[i]=='DR': 
            model.add(Dropout(FILTERS[i]))
            receipt.append('DROP OUT --> '+str(int(100*FILTERS[i]))+' % ' )

    model.add(Dense(2, activation='softmax'))
    receipt.append('OUTPUT --> SOFTMAX 2 CLASSES')

    opt=optimizers.Nadam(lr=LR
                         , beta_1=0.9
                         , beta_2=0.999
                         , epsilon=1e-8
                         , schedule_decay=0.000002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if verbose==True:
        for line in receipt:
            print(line)

    return(model) 
    
#################################################################################################################################
#################################################################################################################################
    
    
################################################################    
## This function trains a model with the given compiled model ##
################################################################
 
def train_model(model               # Model or already saved model from directory
               ,dataset             # Dataset to train with
               ,epoch               # Epochs of training
               ,batch               # Batch size of the training 
               ,split               # Split ratio of TEST / TRAINING data
               ,classes=2           # Number of classes used in this training
               ,save_model=False  # (optional) I you want to save the model, assign name
               ,data_source_path='/home/vasileios.skliris/EMILY/datasets/'
               ,model_source_path='/home/vasileios.skliris/EMILY/trainings/'):

    if isinstance(model,str): 
        model_fin=load_model(model_source_path+model+'.h5')
    else:
        model_fin=model
        
    if isinstance(dataset,str):
        data = io.loadmat(data_source_path+dataset+'.mat')
        X = data['data']
        Y = data['labels']

        Y = np_utils.to_categorical(Y, classes) # [0] --> [1 0] = NOISE if classes=2
                                                # [1] --> [0 1] = SIGNAL
        print('... Loading file '+dataset+' with data shape:  ',X.shape)
    
    else:
        X, Y = dataset[0], dataset[1]
        Y = np_utils.to_categorical(Y, classes)
        print('... Loading dataset with data shape:  ',X.shape)

    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=split,random_state=0)
    print('... Spliting the dada with ratio: ',split)
    print('... and final shapes of: ',X_train.shape, X_test.shape)
    

    hist=model_fin.fit(X_train
                   ,Y_train
                   ,epochs=epoch
                   ,batch_size=batch
                   ,validation_data=(X_test, Y_test))
    if isinstance(save_model,str):
        model_fin.save(save_model+'.h5')
    return(hist.history)


#################################################################################################################################
#################################################################################################################################


########################################################
## A function to import a model and test in selected  ##
## dataset. Just use name of dataset. Modify the path ##
## for personal use.                                  ##
########################################################

def test_model(model              # Model to test
               ,test_data         # Testing dateset
               ,extended=False    # Printing extra stuff   NEEDS DEBUGIN
               ,classes = 2       # Number of classes
               ,data_source_path='/home/vasileios.skliris/EMILY/datasets/'
               ,model_source_path='/home/vasileios.skliris/EMILY/trainings/'):
    
    if isinstance(model,str):
        trained_model = load_model(model_source_path+ model +'.h5')
    else:
        trained_model = model    #If model is not already in the script you import it my calling the name
    
    if isinstance(test_data,str):
        data = io.loadmat(data_source_path+test_data+'.mat')
        X = data['data']
        Y = data['labels']
        
        #Y_categ = np_utils.to_categorical(Y, 2) # [0] --> [1 0] = NOISE
                                                 # [1] --> [0 1] = SIGNAL       
    else:
        X, Y = test_data[0], test_data[1]
        Y = np_utils.to_categorical(Y, classes)
        print('... Loading dataset with data shape:  ',X.shape)
        

    
    # Confidence that the data is [NOISE | SIGNAL] = [0 1] = [[1 0],[0 1]]
    predictions=trained_model.predict_proba(X, batch_size=1, verbose=1)
    
    #scores=predictions[predictions[:,0].argsort()] 
    
    if extended==True:
        pr1, pr0 = [], []
        true_labels=[]

        for i in range(0,len(Y)):
            if Y[i][1]==1:
                true_labels.append('SIGNAL')
                pr1.append(predictions[i])

            elif Y[i][1]==0:
                true_labels.append('NOISE')
                pr0.append(predictions[i])
        
                       
        pr0, pr1 = np.array(pr0), np.array(pr1)

        plt.ioff()
        plt.figure(figsize=(15,10))
        if len(pr0)!=0:
            n0, b0, p0 = plt.hist(pr0[:,1],bins=1000,log=True,cumulative=-1,color='r',alpha=0.5)
        if len(pr1)!=0:
            n1, b1, p1 = plt.hist(pr1[:,1],bins=1000,log=True,cumulative=1,color='b',alpha=0.5)
        #plt.ylabel('Counts')
        #plt.title('Histogram of '+test_data)
        
        new_name=[]
        for i in test_data[::-1]:
            if i!='/':
                new_name.append(i)
            else:
                break  
        new_name=''.join(new_name[::-1])
        fig_name=new_name+'.png'
        plt.savefig(fig_name)
    
    return(predictions)

##################################################################################################################################################################################################################################################################

def lr_change(model,new_lr):
    old_lr=K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr,new_lr)
    lr=K.get_value(model.optimizer.lr)

    print('Learning rate changed from '+str(old_lr)+' to '+str(lr))
    
    return

##################################################################################################################################################################################################################################################################

def save_history(histories,name='nonamed_history',save=False,extendend=True): 

    train_loss_history = []
    train_acc_history=[]
    val_loss_history = []
    val_acc_history=[]

    for new_history in histories:
        val_loss_history = val_loss_history + new_history['val_loss']
        val_acc_history = val_acc_history + new_history['val_acc'] 
        train_loss_history = train_loss_history + new_history['loss'] 
        train_acc_history = train_acc_history + new_history['acc'] 

    history_total={'val_acc': val_acc_history
                   , 'acc': train_acc_history
                   , 'val_loss':val_loss_history
                   , 'loss': train_loss_history,}
    if save == True:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(history_total, f, pickle.HIGHEST_PROTOCOL)
    
    if extendend == True:
        epochs=np.arange(1,len(history_total['val_acc'])+1)
        
        plt.ioff()
        plt.figure(figsize=(15,10))
        plt.plot(epochs,history_total['acc'],'b')
        plt.plot(epochs,history_total['val_acc'],'r')
        plt.title('Accuracy of validation and testing')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
         
        new_name=[]                      # 
        for i in name[::-1]:             #
            if i!='/':
                new_name.append(i)       # routine to avoid naming errors with paths
            else:
                break                    #
        new_name=''.join(new_name[::-1]) # 
        fig_name=new_name+'.png'         #
        plt.savefig(fig_name)

    return(history_total)

##################################################################################################################################################################################################################################################################


def data_fusion(names     #Takes as input the datasets name [, , , ...] files and combines them to one new.
                ,sizes=None
                ,save=False
                ,data_source_file='/home/vasileios.skliris/EMILY/datasets/'):      
    
    XS,YS=[],[]
    for name in names:
        data = io.loadmat(data_source_file+name+'.mat')
        X = data['data']
        Y = data['labels']
        print('Loading file ...'+name+' with data shape:  ',X.shape)
        XS.append(X)
        YS.append(Y)
        
    if sizes==None:
        data=np.vstack(XS[:])
        labels=np.vstack(YS[:])
    elif len(sizes)==len(names):
        data=np.array(XS[0][:sizes[0]])
        labels=np.array(YS[0][:sizes[0]])
        for i in np.arange(1,len(sizes)):
            data=np.vstack((data,np.array(XS[i][:sizes[i]])))
            labels=np.vstack((labels,np.array(YS[i][:sizes[i]])))
    print(data.shape)
        
    s=np.arange(data.shape[0])
    np.random.shuffle(s)
    
    data=data[s]
    labels=labels[s]
    print('Files were fused with data shape:  ',data.shape , labels.shape)
    
    if isinstance(save,str):
        d={'data': data,'labels': labels}             
        io.savemat('/home/vasileios.skliris/EMILY/datasets/'+save+'.mat',d)
        print('File '+save+'.mat was created')
    else:
        return(data, labels)

################################################################################################################################################################################################################################################################

    
    

    
    
################################################################################
#################### DOCUMENTATION OF data_generator_cbc## #####################
################################################################################
#                                                                              #
# parameters       (list) A list with elemntets the source_cbcs,noise_type and #
#                  the SNR we want the injections to have. Noise_types are:    #
#                                                                              #
#    'optimal'     Generated following the curve of ligo and virgo and followi #
#                  ng simulateddetectornoise.py                                #
#    'sudo_real'   Generated as optimal but the PSD curve is from real data PS #
#                  D.                                                          #
#    'real'        Real noise from the detectors.                              #
#                                                                              #
# length:          (float) The length in seconds of the generated instantiatio #
#                  ns of data.                                                 #
#                                                                              #
# fs:              (int) The sample frequency of the data. Use powers of 2 for #
#                  faster calculations.                                        #
#                                                                              #
# size:            (int) The amound of instantiations you want to generate. Po #
#                  wers of are more convinient.                                #
#                                                                              #
# detectors:       (string) A string with the initial letter of the detector y #
#                  ou want to include. H for LIGO Hanford, L for LIGO Livingst #
#                  on, V for Virgo and K for KAGRA. Example: 'HLV' or 'HLVK' o # 
#                  r 'LV'.                                                     #
#                                                                              #                                   
# spec:            (optional/boolean): The option to also generate spectrogram #
#                  s. Default is false. If true it will generate a separate da #
#                  taset with pictures of spectrograms of size related to the  #
#                  variable res below.                                         #
#                                                                              #
# phase:           (optional/boolean): Additionaly to spectrograms you can gen #
#                  erate phasegrams. Default is false. If true it will generat #
#                  e an additional picture under the spectrograms with the sam #
#                  e size. The size will be the same as spectrogram.           #
#                                                                              #
# res:             NEED MORE INFO HERE.                                        #
#                                                                              #
# noise_file:      (optional if noise_type is 'optimal'/ [str,str]) The name o #
#                  f the real data file you want to use as source. The path is #
#                  setted in load_noise function on emily.py. THIS HAS TO BE   #
#                  FIXED LATER!                                                #
#                  The list includes ['date_folder', 'name_of_file']. Date_fol #
#                  der is inside ligo data and contains segments of that day w #
#                  ith the name witch should be the name_of_file.              #
#                                                                              #
# t:               (optinal except psd_mode='default/ float)The duration of th # 
#                  e envelope of every instantiation used to generate the psd. # 
#                  It is used in psd_mode='default'. Prefered to be integral o #
#                  f power of 2. Default is 32.                                #
#                                                                              #
# batch_size:      (odd int)The number of instantiations we will use for one b #
#                  atch. To make our noise instanstiation indipendent we need  #
#                  a specific number given the detectors. For two detectors gi #
#                  ves n-1 andf or three n(n-1) indipendent instantiations.    #
#                                                                              #
# name:            (optional/string) A special tag you might want to add to yo #
#                  ur saved dataset files. Default is ''.                      #
#                                                                              #
# destination_path:(optional/string) The path where the dataset will be saved, # 
#                  the default is '/home/vasileios.skliris/EMILY/datasets/'    #
#                                                                              #
# demo:            (optional/boolean) An option if you want to have a ploted o #
#                  ovrveiw of the data that were generated. It will not work i #
#                  n command line enviroment. Default is false.                #
################################################################################


def data_generator_cbc(parameters        
                       ,length           
                       ,fs               
                       ,size             
                       ,detectors='HLV'  
                       ,spec=True
                       ,phase=True
                       ,res=128
                       ,noise_file=None  
                       ,t=32             
                       ,batch_size=11
                       ,starting_point=0
                       ,name=''          
                       ,destination_path='/home/vasileios.skliris/EMILY/datasets/cbc/'
                       ,demo=False):       

    ## INPUT CHECKING ##########
    #
    
    # parameters
    if not (isinstance(parameters[0],str) and isinstance(parameters[1],str) and (isinstance(parameters[2],float) or isinstance(parameters[2],int)) and len(parameters)==3):
        raise ValueError('The parameters have to be three and in the form: [list, list , float/int]')
             
    if not (os.path.isdir('/home/vasileios.skliris/EMILY/injections/cbcs/'+parameters[0])):
        raise FileNotFoundError('No such file or directory: \'/home/vasileios.skliris/EMILY/injections/'+parameters[0]) 
    if (parameters[1]!='optimal' and parameters[1]!='sudo_real' and parameters[1]!='real'): 
        raise ValueError('Wrong type of noise, the only acceptable are: \n\'optimal\'\n\'sudo_real\'\n\'real\'')
    if (parameters[2]<0):
        raise ValueError('You cannot have a negative SNR')
    
    #length
    if (not (isinstance(length,float) or isinstance(length,int)) and length>0):
        raise ValueError('The length value has to be a possitive float or integer.')
        
    # fs
    if not isinstance(fs,int) or fs<=0:
        raise ValueError('Sample frequency has to be a positive integer.')
    
    # detectors
    for d in detectors:
        if (d!='H' and d!='L' and d!='V' and d!='K'): 
            raise ValueError('Not acceptable character for detector. You should include: \nH for LIGO Hanford\nL for LIGO Livingston \nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')
    # res
    if not isinstance(res,int):
        raise ValueError('Resolution variable (res) can only be integral')
    
    # noise_file    
    if (parameters[1]=='sudo_real' or parameters[1]=='real') and noise_file==None:
        raise TypeError('If you use suno_real or real noise you need a real noise file as a source.')         
    if noise_file!=None and len(noise_file)==2 and isinstance(noise_file[0],str) and isinstance(noise_file[1],str):
        for d in detectors:
            if os.path.isdir('/home/vasileios.skliris/EMILY/ligo_data/'+str(fs)+'/'+noise_file[0]+'/'+d+'/'+noise_file[1]+'.txt'):
                raise FileNotFoundError('No such file or directory: \'/home/vasileios.skliris/EMILY/ligo_data/'+str(fs)+'/'+noise_file[0]+'/'+d+'/'+noise_file[1]+'.txt\'')
                
    # t
    if not isinstance(t,int):
        raise ValueError('t needs to be an integral')
        
    # batch size:
    if not (isinstance(batch_size, int) and batch_size%2!=0):
        raise ValueError('batch_size has to be an odd integer')

    # name
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')
    
    # destination_path
    if not os.path.isdir(destination_path): 
        raise ValueError('No such path '+destination_path)
    #                        
    ########## INPUT CHECKING ## 
    
    

    dataset = parameters[0]
    noise_type = parameters[1]
    SNR_FIN = parameters[2]

    lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  # Labels used in saving file
    if size not in lab:
        lab[size]=str(size)

    
    # Making a list of the injection names, so that we can sample randomly from them
    if 'H' in detectors: injectionH=dirlist('/home/vasileios.skliris/EMILY/injections/cbcs/'+dataset+'/H')
    if 'L' in detectors: injectionL=dirlist('/home/vasileios.skliris/EMILY/injections/cbcs/'+dataset+'/L')
    if 'V' in detectors: injectionV=dirlist('/home/vasileios.skliris/EMILY/injections/cbcs/'+dataset+'/V')
        
    fl, fm=20, int(fs/2)   # Integration limits for the calculation of analytical SNR

    magic={2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
    param=magic[fs]   # Magic number to mach the analytical computation of SNR and the match filter one:
                      # There was a mis-match which I coulndnt resolve how to fix this
                      # and its not that important, if we get another nobel I will address that.

    DATA=[]
   

        ##########################
        #                        #
        # CASE OF OPTIMAL NOISE  #       
        #                        #
        ##########################
        
    if noise_type=='optimal':
        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL))   
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV))     
            
            if 'H' in detectors:

                PSDH,XH,TH=simulateddetectornoise('aligo',t,fs,10,fs/2)   # Creation of the artificial noise.
                
                injH=load_inj(dataset,injectionH[inj_ind],'H')        # Calling the templates generated with PyCBC
                inj_len=len(injH)/fs                                  # Saving the length of the injection

                if length==inj_len:                                       # I put a random offset for all injection
                    disp = np.random.randint(0,int(length*fs/2))          # so that the signal is not always in the 
                elif length > inj_len:                                    # same place.
                    disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                             ,int(fs*(length-inj_len)/2)) #

                if disp>=0: 
                    injH=injH[disp:]                              # Due to offset the injection file will be
                if disp<0: 
                    injH=injH[0:-disp]  
            
                    
                injH=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injH
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 
                H_back=TimeSeries(XH,sample_rate=fs)                      # Making the noise a TimeSeries
                asdH=H_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later


                injH_fft_0=np.fft.fft(injH)                       # Calculating the one sided fft of the template, 
                injH_fft_0N=2*np.abs(injH_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N*injH_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDH[t*fl-1:t*fm-1]))
                
            if 'L' in detectors:

                PSDL,XL,TL=simulateddetectornoise('aligo',t,fs,10,fs/2)   # Creation of the artificial noise.

                injL=load_inj(dataset,injectionL[inj_ind],'L')                # Calling the templates generated with PyCBC
                inj_len=len(injL)/fs                                  # Saving the length of the injection
                
                if 'H' not in detectors:
                    if length==inj_len:                                   # I put a random offset for all injection
                        disp = np.random.randint(0,int(length*fs/2))      # so that the signal is not always in the 
                    elif length > inj_len:                                    # same place.
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                                 ,int(fs*(length-inj_len)/2)) #
                
                if disp>=0: 
                    injL=injL[disp:]                              # Due to offset the injection file will be
                if disp<0: 
                    injL=injL[0:-disp]  

                injL=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injL
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 

                L_back=TimeSeries(XL,sample_rate=fs)                      # Making the noise a TimeSeries
                asdL=L_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later             
                
                injL_fft_0=np.fft.fft(injL)                       # Calculating the one sided fft of the template, 
                injL_fft_0N=2*np.abs(injL_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N*injL_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:

                PSDV,XV,TV=simulateddetectornoise('avirgo',t,fs,10,fs/2)  # Creation of the artificial noise.

                injV=load_inj(dataset,injectionV[inj_ind],'V')        # Calling the templates generated with PyCBC
                inj_len=len(injV)/fs                                  # Saving the length of the injection
                
                if ('H' and 'L') not in detectors:
                    if length==inj_len:                                   # I put a random offset for all injection
                        disp = np.random.randint(0,int(length*fs/2))      # so that the signal is not always in the 
                    elif length > inj_len:                                    # same place.
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                                 ,int(fs*(length-inj_len)/2)) #
                        
                if disp>=0: 
                    injV=injV[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injV=injV[0:-disp]                     
                injV=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injV
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 


                V_back=TimeSeries(XV,sample_rate=fs)                      # Making the noise a TimeSeries
                asdV=V_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later                


                injV_fft_0=np.fft.fft(injV)                       # Calculating the one sided fft of the template, 
                injV_fft_0N=2*np.abs(injV_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N*injV_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDV[t*fl-1:t*fm-1]))

                
                
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     # Calculation of combined SNR
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)

            if 'H' in detectors:
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         # Tuning injection amplitude to the SNR wanted
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))

                HF=TimeSeries(XH+injH_cal,sample_rate=fs,t0=0)
                h=HF.whiten(1,0.5,asd=asdH)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data


            if 'L' in detectors:

                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         # Tuning injection amplitude to the SNR wanted
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))

                LF=TimeSeries(XL+injL_cal,sample_rate=fs,t0=0)
                l=LF.whiten(1,0.5,asd=asdL)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data

            if 'V' in detectors:

                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         # Tuning injection amplitude to the SNR wanted
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))

                VF=TimeSeries(XV+injV_cal,sample_rate=fs,t0=0)
                v=VF.whiten(1,0.5,asd=asdV)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print('CBC',i) 
            
            
            
        ############################
        #                          #
        # CASE OF SUDO-REAL NOISE  #       
        #                          #
        ############################
            
    if noise_type=='sudo_real':
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0],'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0],'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0],'V',noise_file[1])
        
        ind=index_combinations(detectors = detectors
                               ,inst = batch_size
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL))   
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV))
                
            if 'H' in detectors:

                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseH, Fs=fs, NFFT=fs, visible=False) # Generating the PSD of it
                p, f=p[1::],f[1::]
                
                PSDH,XH,TH=simulateddetectornoise([f,p],t,fs,10,fs/2)     # Feeding the PSD to generate the sudo-real noise.            

                injH=load_inj(dataset,injectionH[inj_ind],'H')        # Calling the templates generated with PyCBC
                inj_len=len(injH)/fs                                  # Saving the length of the injection
                
                if length==inj_len:                                       # I put a random offset for all injection
                    disp = np.random.randint(0,int(length*fs/2))          # so that the signal is not always in the 
                elif length > inj_len:                                    # same place.
                    disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                             ,int(fs*(length-inj_len)/2)) #

                if disp>=0: 
                    injH=injH[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injH=injH[0:-disp]                     
                injH=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injH
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 

                H_back=TimeSeries(XH,sample_rate=fs)                      # Making the noise a TimeSeries
                asdH=H_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later


                injH_fft_0=np.fft.fft(injH)                       # Calculating the one sided fft of the template, 
                injH_fft_0N=2*np.abs(injH_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N*injH_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDH[t*fl-1:t*fm-1]))

            if 'L' in detectors:

                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseL, Fs=fs, NFFT=fs, visible=False) # Generating the PSD of it
                p, f=p[1::],f[1::]
                
                PSDL,XL,TL=simulateddetectornoise([f,p],t,fs,10,fs/2)     # Feeding the PSD to generate the sudo-real noise.
                
                injL=load_inj(dataset,injectionL[inj_ind],'L')        # Calling the templates generated with PyCBC
                inj_len=len(injL)/fs                                  # Saving the length of the injection
                
                if 'H' not in detectors:
                    if length==inj_len:                                   # I put a random offset for all injection
                        disp = np.random.randint(0,int(length*fs/2))      # so that the signal is not always in the 
                    elif length > inj_len:                                    # same place.
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                                 ,int(fs*(length-inj_len)/2)) #

                if disp>=0: 
                    injL=injL[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injL=injL[0:-disp]                     
                injL=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injL
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 
                L_back=TimeSeries(XL,sample_rate=fs)                      # Making the noise a TimeSeries
                asdL=L_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later             
                
                injL_fft_0=np.fft.fft(injL)                       # Calculating the one sided fft of the template, 
                injL_fft_0N=2*np.abs(injL_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N*injL_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:
                
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseV, Fs=fs, NFFT=fs, visible=False) # Generating the PSD of it
                p, f=p[1::],f[1::]
                
                PSDV,XV,TV=simulateddetectornoise([f,p],t,fs,10,fs/2)     # Feeding the PSD to generate the sudo-real noise.  
                
                injV=load_inj(dataset,injectionV[inj_ind],'V')        # Calling the templates generated with PyCBC
                inj_len=len(injV)/fs                                  # Saving the length of the injection
                
                if ('H' and 'L') not in detectors:
                    if length==inj_len:                                   # I put a random offset for all injection
                        disp = np.random.randint(0,int(length*fs/2))      # so that the signal is not always in the 
                    elif length > inj_len:                                    # same place.
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                                 ,int(fs*(length-inj_len)/2)) #
                        
                if disp>=0: 
                    injV=injV[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injV=injV[0:-disp]                     
                injV=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injV
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 


                V_back=TimeSeries(XV,sample_rate=fs)                      # Making the noise a TimeSeries
                asdV=V_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later                


                injV_fft_0=np.fft.fft(injV)                       # Calculating the one sided fft of the template, 
                injV_fft_0N=2*np.abs(injV_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N*injV_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDV[t*fl-1:t*fm-1]))

                
                
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     # Calculation of combined SNR
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)

            if 'H' in detectors:
                
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         # Tuning injection amplitude to the SNR wanted
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))

                HF=TimeSeries(XH+injH_cal,sample_rate=fs,t0=0)
                h=HF.whiten(1,0.5,asd=asdH)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data
                
            if 'L' in detectors:

                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         # Tuning injection amplitude to the SNR wanted
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))

                LF=TimeSeries(XL+injL_cal,sample_rate=fs,t0=0)
                l=LF.whiten(1,0.5,asd=asdL)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data

            if 'V' in detectors:

                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         # Tuning injection amplitude to the SNR wanted
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))

                VF=TimeSeries(XV+injV_cal,sample_rate=fs,t0=0)
                v=VF.whiten(1,0.5,asd=asdV)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print('CBC',i)
            

        #######################
        #                     #
        # CASE OF REAL NOISE  #       
        #                     #
        #######################

    if noise_type=='real':
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0],'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0],'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0],'V',noise_file[1])
       
        ind=index_combinations(detectors = detectors
                               ,inst= batch_size
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)



        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL))   
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV)) 
                
                
            if 'H' in detectors:
                
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseH, Fs=fs,NFFT=fs)                 # Calculatint the psd of FFT=1s
                psd_int=interp1d(f,p)                                     # Interpolate so that has t*fs values
                PSDH=psd_int(np.arange(0,fs/2,1/t))
                
                injH=load_inj(dataset,injectionH[inj_ind],'H')        # Calling the templates generated with PyCBC
                inj_len=len(injH)/fs                                  # Saving the length of the injection
                
                if length==inj_len:                                       # I put a random offset for all injection
                    disp = np.random.randint(0,int(length*fs/2))          # so that the signal is not always in the 
                elif length > inj_len:                                    # same place.
                    disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                             ,int(fs*(length-inj_len)/2)) #

                if disp>=0: 
                    injH=injH[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injH=injH[0:-disp]                     
                injH=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injH
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 

                H_back=TimeSeries(noiseH,sample_rate=fs)              # Making the noise a TimeSeries
                asdH=H_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later


                injH_fft_0=np.fft.fft(injH)                       # Calculating the one sided fft of the template, 
                injH_fft_0N=2*np.abs(injH_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.
                
                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N*injH_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDH[t*fl-1:t*fm-1]))
                
            if 'L' in detectors:

                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseL, Fs=fs,NFFT=fs)                 # Calculatint the psd of FFT=1s
                psd_int=interp1d(f,p)                                     # Interpolate so that has t*fs values
                PSDL=psd_int(np.arange(0,fs/2,1/t))
        
                injL=load_inj(dataset,injectionL[inj_ind],'L')        # Calling the templates generated with PyCBC
                inj_len=len(injL)/fs                                  # Saving the length of the injection
                
                if 'H' not in detectors:
                    if length==inj_len:                                   # I put a random offset for all injection
                        disp = np.random.randint(0,int(length*fs/2))      # so that the signal is not always in the 
                    elif length > inj_len:                                    # same place.
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                                 ,int(fs*(length-inj_len)/2)) #

                if disp>=0: 
                    injL=injL[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injL=injL[0:-disp]                     
                injL=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injL
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))   # Zero-padding the injection 
                L_back=TimeSeries(noiseL,sample_rate=fs)              # Making the noise a TimeSeries
                asdL=L_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later             
                
                injL_fft_0=np.fft.fft(injL)                       # Calculating the one sided fft of the template, 
                injL_fft_0N=2*np.abs(injL_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N*injL_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:

                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]           # Calling the real noise segments
                
                print(len(noiseV),ind['V'][i]/fs,t,fs)   

                
                p, f = plt.psd(noiseV, Fs=fs,NFFT=fs)                 # Calculatint the psd of FFT=1s
                psd_int=interp1d(f,p)                                     # Interpolate so that has t*fs values
                PSDV=psd_int(np.arange(0,fs/2,1/t))
                
                injV=load_inj(dataset,injectionV[inj_ind],'V')        # Calling the templates generated with PyCBC
                inj_len=len(injV)/fs                                  # Saving the length of the injection
                
                if ('H' and 'L') not in detectors:
                    if length==inj_len:                                   # I put a random offset for all injection
                        disp = np.random.randint(0,int(length*fs/2))      # so that the signal is not always in the 
                    elif length > inj_len:                                    # same place.
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)  # 
                                                 ,int(fs*(length-inj_len)/2)) #
                        
                if disp>=0: 
                    injV=injV[disp:]                              # Due to offset the injection file will be

                if disp<0: 
                    injV=injV[0:-disp]                     
                injV=np.hstack((np.zeros(int(fs*(t-inj_len)/2))
                                ,injV
                                ,np.zeros(int(fs*(t-inj_len)/2+disp))))  # Zero-padding the injection 


                V_back=TimeSeries(noiseV,sample_rate=fs)              # Making the noise a TimeSeries
                asdV=V_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later                


                injV_fft_0=np.fft.fft(injV)                       # Calculating the one sided fft of the template, 
                injV_fft_0N=2*np.abs(injV_fft_0[1:int(t*fs/2)+1]) # we multiply by two and we get rid of the DC value 
                                                                          # and everything above fs/2.

                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N*injV_fft_0N.conjugate())[t*fl-1:t*fm-1]/PSDV[t*fl-1:t*fm-1]))

            print(len(noiseV),ind['V'][i]/fs,t,fs)   
                
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     # Calculation of combined SNR
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)

            #plt.figure()            
            
            if 'H' in detectors:

                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         # Tuning injection amplitude to the SNR wanted
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))

                HF=TimeSeries(noiseH+injH_cal,sample_rate=fs,t0=0)
                h=HF.whiten(1,0.5,asd=asdH)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data

            if 'L' in detectors:

                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         # Tuning injection amplitude to the SNR wanted
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                
                LF=TimeSeries(noiseL+injL_cal,sample_rate=fs,t0=0)
                l=LF.whiten(1,0.5,asd=asdL)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data
                

            if 'V' in detectors:

                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         # Tuning injection amplitude to the SNR wanted
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))

                VF=TimeSeries(noiseV+injV_cal,sample_rate=fs,t0=0)
                v=VF.whiten(1,0.5,asd=asdV)#[int(((t-length)/2)*fs):int(((t+length)/2)*fs)] #Whitening final data
            

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print('CBC',i,)       
                
    # We are gonna use Keras which demands a specific format of datasets
    # So we do that formating here
    
    data=[]
    labels=[]
    for row in DATA:
        data.append(row[0])
        labels=labels+[row[1]]
        
    del DATA   
    data, labels= np.array(data), np.array(labels)
    labels=labels.reshape(1*size,1)

    
    data = data.transpose((0,2,1))
    
    print('Shape of data created:   ',data.shape)
    print('Shape of labels created: ',labels.shape)
    
    if demo==True:
        for i in range(0,10):
            plt.figure(figsize=(10,5))
            
            for j in range(0,len(detectors)):
                plt.plot(np.arange(0,length,1./fs),data[i][:,j]+j*4)
                
            plt.show()


    d={'data': data,'labels': labels}
    save_name=detectors+'_time_cbc_with_'+noise_type+'_noise_SNR'+str(SNR_FIN)+'_'+name+lab[size]
    print(destination_path+save_name)
    io.savemat(destination_path+save_name,d)
    print('File '+save_name+'.mat was created')
    
    if spec==True:
        
        data_spec=[]

        #for time_series in data:
        #    spec=[]
        #    for i in range(0,len(detectors)):
        #        ts=TimeSeries(time_series[:,i],sample_rate=fs,t0=0)
        #        spec_gram = ts.spectrogram(stride, fft_l) ** (1/2.)
        #        spectogram=np.array(spec_gram)
        #        spec.append(spectogram)
        #    data_spec.append(spec)
        # data_spec = data_spec.transpose((0,3,2,1))

        stride ,fft_l = res/fs, res/fs  

        for time_series in data:
            spec=[]
            for i in range(0,len(detectors)):
                f, t, spec_gram = spectrogram(time_series[:,i]
                            , fs 
                            , window='hanning'
                            , nperseg=int(stride*fs)
                            , noverlap=0
                            , nfft =int(fft_l*fs)
                            , mode='complex'
                            ,scaling ='density')
                spectogram=np.array(np.abs(spec_gram))
                if phase==True: phasegram=np.array(np.angle(spec_gram))
                spec.append(spectogram)
                if phase==True: spec.append(phasegram)

            data_spec.append(spec)

        data_spec=np.array(data_spec)

        data_spec = data_spec.transpose((0,2,3,1))
        #data_spec = np.flip(data_spec,1)    

        print('Shape of data created:   ',data_spec.shape)
        print('Shape of labels created: ',labels.shape)

        d={'data': data_spec,'labels': labels}        
        save_name=detectors+'_spec_cbc_with_'+noise_type+'_noise_SNR'+str(SNR_FIN)+'_'+name+lab[size]
        io.savemat(destination_path+save_name,d)
        print('File '+save_name+'.mat was created')
    
        if demo==True:
            for i in range(0,10):
                if len(detectors)==3:
                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,::2]/np.max(data_spec[i][:,:,::2])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1::2]/np.max(data_spec[i][:,:,1::2])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i]/np.max(data_spec[i]),extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                elif len(detectors) in [1,2]:

                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,0]/np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1]/np.max(data_spec[i][:,:,1])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i][:,:,0]/np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')


            plt.show()

            
################################################################################
#################### DOCUMENTATION OF data_generator_noise #####################
################################################################################
#                                                                              #
# noise_type       (string) The type of noise to be generated. There are three #
#                  options:                                                    #
#                                                                              #
#    'optimal'     Generated following the curve of ligo and virgo and followi #
#                  ng simulateddetectornoise.py                                #
#    'sudo_real'   Generated as optimal but the PSD curve is from real data PS #
#                  D.                                                          #
#    'real'        Real noise from the detectors.                              #
#                                                                              #
# length:          (float) The length in seconds of the generated instantiatio #
#                  ns of data.                                                 #
#                                                                              #
# fs:              (int) The sample frequency of the data. Use powers of 2 for #
#                  faster calculations.                                        #
#                                                                              #
# size:            (int) The amound of instantiations you want to generate. Po #
#                  wers of are more convinient.                                #
#                                                                              #
# detectors:       (string) A string with the initial letter of the detector y #
#                  ou want to include. H for LIGO Hanford, L for LIGO Livingst #
#                  on, V for Virgo and K for KAGRA. Example: 'HLV' or 'HLVK' o # 
#                  r 'LV'.                                                     #
#                                                                              #                                   
# spec:            (optional/boolean): The option to also generate spectrogram #
#                  s. Default is false. If true it will generate a separate da #
#                  taset with pictures of spectrograms of size related to the  #
#                  variable res below.                                         #
#                                                                              #
# phase:           (optional/boolean): Additionaly to spectrograms you can gen #
#                  erate phasegrams. Default is false. If true it will generat #
#                  e an additional picture under the spectrograms with the sam #
#                  e size. The size will be the same as spectrogram.           #
#                                                                              #
# res:             NEED MORE INFO HERE.                                        #
#                                                                              #
# noise_file:      (optional if noise_type is 'optimal'/ [str,str]) The name o #
#                  f the real data file you want to use as source. The path is #
#                  setted in load_noise function on emily.py. THIS HAS TO BE   #
#                  FIXED LATER!                                                #
#                  The list includes ['date_folder', 'name_of_file']. Date_fol #
#                  der is inside ligo data and contains segments of that day w #
#                  ith the name witch should be the name_of_file.              #
#                                                                              #
# t:               (optinal except psd_mode='default/ float)The duration of th # 
#                  e envelope of every instantiation used to generate the psd. # 
#                  It is used in psd_mode='default'. Prefered to be integral o #
#                  f power of 2. Default is 32.                                #
#                                                                              #
# batch_size:      (odd int)The number of instantiations we will use for one b #
#                  atch. To make our noise instanstiation indipendent we need  #
#                  a specific number given the detectors. For two detectors gi #
#                  ves n-1 andf or three n(n-1) indipendent instantiations.    #
#                                                                              #
# name:            (optional/string) A special tag you might want to add to yo #
#                  ur saved dataset files. Default is ''.                      #
#                                                                              #
# destination_path:(optional/string) The path where the dataset will be saved, # 
#                  the default is '/home/vasileios.skliris/EMILY/datasets/'    #
#                                                                              #
# demo:            (optional/boolean) An option if you want to have a ploted o #
#                  ovrveiw of the data that were generated. It will not work i #
#                  n command line enviroment. Default is false.                #
################################################################################


def data_generator_noise(noise_type         
                         ,length            
                         ,fs                
                         ,size              
                         ,detectors='HLV'   
                         ,spec=False        
                         ,phase=False       
                         ,res=128           
                         ,noise_file=None   
                         ,t=32              
                         ,batch_size=1
                         ,starting_point=0
                         ,name=''           
                         ,destination_path='/home/vasileios.skliris/EMILY/datasets/noise/'
                         ,demo=False):      
    
    ## INPUT CHECKING ##########
    #
    
    # noise_type
    if (noise_type!='optimal' and noise_type!='sudo_real' and noise_type!='real'): 
        raise ValueError('Wrong type of noise, the only acceptable are: \n\'optimal\'\n\'sudo_real\'\n\'real\'')
    
    #length
    if (not (isinstance(length,float) or isinstance(length,int)) and length>0):
        raise ValueError('The length value has to be a possitive float or integer.')
        
    # fs
    if not isinstance(fs,int) or fs<=0:
        raise ValueError('Sample frequency has to be a positive integer.')
    
    # detectors
    for d in detectors:
        if (d!='H' and d!='L' and d!='V' and d!='K'): 
            raise ValueError('Not acceptable character for detector. You should include: \nH for LIGO Hanford\nL for LIGO Livingston \nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')
    # res
    if not isinstance(res,int):
        raise ValueError('Resolution variable (res) can only be integral')
    
    # noise_file    
    if (noise_type=='sudo_real' or noise_type=='real') and noise_file==None:
        raise TypeError('If you use suno_real or real noise you need a real noise file as a source.')         
    if noise_file!=None and len(noise_file)==2 and isinstance(noise_file[0],str) and isinstance(noise_file[1],str):
        for d in detectors:
            if os.path.isdir('/home/vasileios.skliris/EMILY/ligo_data/'+str(fs)+'/'+noise_file[0]+'/'+d+'/'+noise_file[1]+'.txt'):
                raise FileNotFoundError('No such file or directory: \'/home/vasileios.skliris/EMILY/ligo_data/'+str(fs)+'/'+noise_file[0]+'/'+d+'/'+noise_file[1]+'.txt\'')
                
    # t
    if not isinstance(t,int):
        raise ValueError('t needs to be an integral')
        
    # batch size:
    if not (isinstance(batch_size, int) and batch_size%2!=0):
        raise ValueError('batch_size has to be an odd integer')

    # name
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')
    
    # destination_path
    if not os.path.isdir(destination_path): 
        raise ValueError('No such path '+destination_path)
    #                        
    ########## INPUT CHECKING ## 

    lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  # Labels used in saving file
    if size not in lab:
        lab[size]=str(size)    
    
    DATA=[]
    
    ##########################
    #                        #
    # CASE OF OPTIMAL NOISE  #       
    #                        #
    ##########################
    if noise_type=='optimal':
        for i in range(0,size):
            
            if 'H'in detectors:
                
                PSDH,XH,TH=simulateddetectornoise('aligo',t,fs,10,fs/2)   # Creation of the artificial noise.            
                H_back=TimeSeries(XH,sample_rate=fs)                      # Making the noise a TimeSeries
                asdH=H_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later
                h=H_back.whiten(1,0.5,asd=asdH)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)]#Whitening final data

            if 'L'in detectors:

                PSDL,XL,TL=simulateddetectornoise('aligo',t,fs,10,fs/2)   # Creation of the artificial noise.            
                L_back=TimeSeries(XL,sample_rate=fs)                      # Making the noise a TimeSeries
                asdL=L_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later
                l=L_back.whiten(1,0.5,asd=asdL)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)]#Whitening final data

            if 'V'in detectors:

                PSDV,XV,TV=simulateddetectornoise('avirgo',t,fs,10,fs/2)  # Creation of the artificial noise.            
                V_back=TimeSeries(XV,sample_rate=fs)                      # Making the noise a TimeSeries
                asdV=V_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later                
                v=V_back.whiten(1,0.5,asd=asdV)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)]#Whitening final data

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))

            row=[dumie,0]  # 'NOISE' = 0 
            DATA.append(row)    
            print('N',i)
                    
            
    ############################
    #                          #
    # CASE OF SUDO-REAL NOISE  #       
    #                          #
    ############################        
    if noise_type =='sudo_real':
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0],'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0],'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0],'V',noise_file[1])
    
        ind=index_combinations(detectors = detectors
                               ,inst = batch_size
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        
        for i in range(0,size):
            
            if 'H'in detectors:
                
                
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseH, Fs=fs, NFFT=fs, visible=False)    # Generating the PSD of it
                p, f=p[1::],f[1::]
                
                PSDH,XH,TH=simulateddetectornoise([f,p],t,fs,10,fs/2)     # Feeding the PSD to generate the sudo-real noise.            
                H_back=TimeSeries(XH,sample_rate=fs)                      # Making the noise a TimeSeries
                asdH=H_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later
                h=H_back.whiten(1,0.5,asd=asdH)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)] #Whitening final data

            if 'L'in detectors:
                
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseL, Fs=fs, NFFT=fs, visible=False)     # Generating the PSD of it
                p, f=p[1::],f[1::]
                
                PSDL,XL,TL=simulateddetectornoise([f,p],t,fs,10,fs/2)     # Feeding the PSD to generate the sudo-real noise.            
                L_back=TimeSeries(XL,sample_rate=fs)                      # Making the noise a TimeSeries
                asdL=L_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later
                l=L_back.whiten(1,0.5,asd=asdL)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)] #Whitening final data

            if 'V'in detectors:

                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]           # Calling the real noise segments
                p, f = plt.psd(noiseV, Fs=fs, NFFT=fs, visible=False)     # Generating the PSD of it
                p, f=p[1::],f[1::]

                PSDV,XV,TV=simulateddetectornoise([f,p],t,fs,10,fs/2)     # Feeding the PSD to generate the sudo-real noise.            
                V_back=TimeSeries(XV,sample_rate=fs)                      # Making the noise a TimeSeries
                asdV=V_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later                
                v=V_back.whiten(1,0.5,asd=asdV)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)] #Whitening final data
            

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))

            row=[dumie,0]  # 'NOISE' = 0 
            DATA.append(row)    
            print('N',i)
            
            
    #######################
    #                     #
    # CASE OF REAL NOISE  #       
    #                     #
    #######################
    if noise_type =='real':
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0],'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0],'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0],'V',noise_file[1])
        
        ind=index_combinations(detectors = detectors
                               ,inst = batch_size
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        print(ind)
        
        for i in range(0,size):


            if 'H'in detectors:
                
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]           # Calling the real noise segments
                H_back=TimeSeries(noiseH,sample_rate=fs)                  # Making the noise a TimeSeries
                asdH=H_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later                
                h=H_back.whiten(1,0.5,asd=asdH)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)] # Whitening data

            if 'L'in detectors:

                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]           # Calling the real noise segments
                L_back=TimeSeries(noiseL,sample_rate=fs)                  # Making the noise a TimeSeries
                asdL=L_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for 
                                                                          # whitening later                
                l=L_back.whiten(1,0.5,asd=asdL)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)] # Whitening data


            if 'V'in detectors:

                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]           # Calling the real noise segments
                V_back=TimeSeries(noiseV,sample_rate=fs)                  # Making the noise a TimeSeries
                asdV=V_back.asd(1,0.5)                                    # Calculating the ASD so tha we can use it for
                                                                          # whitening later                
                v=V_back.whiten(1,0.5,asd=asdV)#[int(((t-length)/2-1)*fs):int(((t+length)/2-1)*fs)] # Whitening data


            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))

            row=[dumie,0]  # 'NOISE' = 0 
            DATA.append(row)    
            print('N',i , ind['H'][i]/fs)  
                    
            
    # We are gonna use Keras which demands a specific format of datasets
    # So we do that formating here

    data=[]
    labels=[]
    for row in DATA:
        data.append(row[0])
        labels=labels+[row[1]]

    del DATA   
    data, labels= np.array(data), np.array(labels)
    
    
    labels=labels.reshape(1*size,1)
    print(data.shape,labels.shape)


    data = data.transpose((0,2,1))

    print('Shape of data created:   ',data.shape)
    print('Shape of labels created: ',labels.shape)

    if demo==True:
        for i in range(0,10):
            plt.figure(figsize=(10,5))
            
            for j in range(0,len(detectors)):
                plt.plot(np.arange(0,length,1./fs),data[i][:,j]+j*4)
            plt.show()

    
    d={'data': data,'labels': labels}
    save_name=detectors+'_time_'+noise_type+'_noise'+'_'+name+lab[size]
    io.savemat(destination_path+save_name,d)
    print('File '+save_name+'.mat was created')

    
    if spec==True:
        data_spec=[]
        stride ,fft_l = res/fs, res/fs  

        for time_series in data:
            spec=[]
            for i in range(0,len(detectors)):
                f, t, spec_gram = spectrogram(time_series[:,i]         # Calculating complex spectrogram
                            , fs 
                            , window='hanning'
                            , nperseg=int(stride*fs)
                            , noverlap=0
                            , nfft =int(fft_l*fs)
                            , mode='complex'
                            ,scaling ='density')
                spectogram=np.array(np.abs(spec_gram))                  
                if phase==True: phasegram=np.array(np.angle(spec_gram)) # Calculating phase
                spec.append(spectogram)
                if phase==True: spec.append(phasegram)                  

            data_spec.append(spec)

        data_spec=np.array(data_spec)

        data_spec = data_spec.transpose((0,2,3,1))
        #data_spec = np.flip(data_spec,1)    

        print('Shape of data created:   ',data_spec.shape)
        print('Shape of labels created: ',labels.shape)

        d={'data': data_spec,'labels': labels}
        save_name=detectors+'_spec_'+noise_type+'_noise'+'_'+name+lab[size]
        io.savemat(destination_path+save_name,d)
        print('File '+save_name+'.mat was created')

        if demo==True:
            for i in range(0,20):
                if len(detectors)==3:
                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,::2]/np.max(data_spec[i][:,:,::2])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1::2]/np.max(data_spec[i][:,:,1::2])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i]/np.max(data_spec[i]),extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                elif len(detectors) in [1,2]:

                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,0]/np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1]/np.max(data_spec[i][:,:,1])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i][:,:,0]/np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')


            plt.show()

