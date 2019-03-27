import sys
sys.path.append('/home/vasileios.skliris/EMILY')


from emily import*

CORE =     ['C','C','C','C','C','C','F','D','DR','D','DR']
MAX_POOL = [  0,  8,  0,  6,  0,  4, 0 ,  0,  0 ,  0,  0 ]
FILTERS =  [ 16, 16, 32, 32, 64, 64, 0 ,128, 0.5, 64, 0.5]
K_SIZE =   [128, 64, 64, 32, 32, 32, 0 ,  0,  0 ,  0,  0 ]

in_shape = (8192,3)
lr = 0.0001



PM=[CORE,MAX_POOL,FILTERS,K_SIZE]

model = conv_model_1D(parameter_matrix = PM
              , INPUT_SHAPE = in_shape 
              , LR = lr
              , verbose=True)
ntype='real'

CBC=['cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR60_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR40_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR30_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR25_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR20_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR16_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR12_XM',
     'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR10_XM']
     #'cbc/cbc_real11/HLV_time_cbc_with_'+ntype+'_noise_SNR8_XM']
NOISE=['noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No1_XM',    
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No2_XM',
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No3_XM',
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No4_XM',
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No5_XM',
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No6_XM',
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No7_XM',
       'noise/'+ntype+'/real11/HLV_time_'+ntype+'_noise_No8_XM']
      # 'noise/'+ntype+'/real5/HLV_time_'+ntype+'_noise_No9_XM']


historia=[]

# THIS IS THE COMPLETE FORM OF THE FIRST RUNNING PROCEDURE

# historia.append(train_model(model = model             # Model or already saved model from directory
#                             ,dataset = data_fusion([CBC[0],NOISE[0]],[10000,10000])              
#                             ,epoch = 80               # Epochs of training
#                             ,batch = 500              # Batch size of the training 
#                             ,split = 0.1              # Split ratio of TEST / TRAINING data
#                             ,save_model=False         # (optional) I you want to save the model, assign name
#                             ,data_source_path='/home/vasileios.skliris/EMILY/datasets/'
#                             ,model_source_path='/home/vasileios.skliris/EMILY/trainings/'))

         
historia.append(train_model(model
                            ,data_fusion([CBC[0],NOISE[0]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR60'))

historia.append(train_model(model
                            ,data_fusion([CBC[1],NOISE[1]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR40'))

historia.append(train_model(model
                            ,data_fusion([CBC[2],NOISE[2]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR30')) 
historia.append(train_model(model
                            ,data_fusion([CBC[3],NOISE[3]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR25'))
historia.append(train_model(model
                            ,data_fusion([CBC[4],NOISE[4]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR20'))
historia.append(train_model(model
                            ,data_fusion([CBC[5],NOISE[5]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR16'))
historia.append(train_model(model
                            ,data_fusion([CBC[6],NOISE[6]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR12'))
historia.append(train_model(model
                            ,data_fusion([CBC[7],NOISE[7]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR10')) 
historia.append(train_model(model
                            ,data_fusion([CBC[8],NOISE[8]],[10000,10000])
                            , 50, 500, 0.1, save_model=ntype+'toSNR8')) 



save_history(historia,name=ntype+'HISTORYtoSNR8_bigger_lr',save=True,extendend=True)


#scores =test_model(model = model                 # Model to test
#                   ,test_data = 'H_TIME_OPT_SNR_4_XM'   # Testing dateset
#                   ,extended=True)               # Printing extra stuff 

# scores =test_model(model = model,test_data = data_fusion(CBC[7],NOISE[7]),extended=True)     #          




                
                   
