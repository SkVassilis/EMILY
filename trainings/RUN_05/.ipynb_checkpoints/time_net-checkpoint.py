import sys
sys.path.append('/home/vasileios.skliris/EMILY')


from emily import*

CORE =     ['C','C','C','C','C','C','F','D','DR','D','DR']
MAX_POOL = [  0,  8,  0,  6,  0,  4, 0 ,  0,  0 ,  0,  0 ]
FILTERS =  [ 16, 16, 32, 32, 64, 64, 0 ,128, 0.5, 64, 0.5]
K_SIZE =   [128, 64, 64, 32, 32, 32, 0 ,  0,  0 ,  0,  0 ]


PM=[CORE,MAX_POOL,FILTERS,K_SIZE]
in_shape = (8192,3)

lr = 0.0001

model = conv_model_1D(parameter_matrix = PM
              , INPUT_SHAPE = in_shape 
              , LR = lr
              , verbose=True)
ntype='real'

CBC=['cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR20_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR18_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR16_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR14_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR12_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR10_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR8_XM',
     'cbc/cbc01/HLV_time_cbc_with_'+ntype+'_noise_SNR6_XM']
NOISE=['noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No1_XM',    
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No2_XM',
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No3_XM',
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No4_XM',
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No5_XM',
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No6_XM',
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No7_XM',
       'noise/'+ntype+'/real01/HLV_time_'+ntype+'_noise_No8_XM']

historia=[]
historia.append(train_model(model = model              # Model or already saved model from directory
                            ,dataset = data_fusion([CBC[0],NOISE[0]],[10000,10000])              
                            ,epoch = 80               # Epochs of training
                            ,batch = 500              # Batch size of the training 
                            ,split = 0.1              # Split ratio of TEST / TRAINING data
                            ,save_model=False         # (optional) I you want to save the model, assign name
                            ,data_source_path='/home/vasileios.skliris/EMILY/datasets/'
                            ,model_source_path='/home/vasileios.skliris/EMILY/trainings/'))


historia.append(train_model(model,data_fusion([CBC[1],NOISE[1]],[10000,10000]), 80, 500, 0.1, save_model=False))
historia.append(train_model(model,data_fusion([CBC[2],NOISE[2]],[10000,10000]), 80, 500, 0.1, save_model=False)) 
historia.append(train_model(model,data_fusion([CBC[3],NOISE[3]],[10000,10000]), 80, 500, 0.1, save_model=ntype+'hlv_30TO10_2_toSNR14'))
historia.append(train_model(model,data_fusion([CBC[4],NOISE[4]],[10000,10000]), 80, 500, 0.1, save_model=ntype+'hlv_30TO10_2_toSNR12'))
historia.append(train_model(model,data_fusion([CBC[5],NOISE[5]],[10000,10000]), 80, 500, 0.1, save_model=ntype+'hlv_30TO10_2_toSNR10'))
historia.append(train_model(model,data_fusion([CBC[6],NOISE[6]],[10000,10000]), 80, 500, 0.1, save_model=ntype+'hlv_30TO10_2_toSNR8'))
historia.append(train_model(model,data_fusion([CBC[7],NOISE[7]],[10000,10000]), 80, 500, 0.1, save_model=ntype+'hlv_30TO10_2_toSNR6')) 



save_history(historia,name=ntype+'hlv_30TO10_2_toSNR6',save=True,extendend=True)


#scores =test_model(model = model                 # Model to test
#                   ,test_data = 'H_TIME_OPT_SNR_4_XM'   # Testing dateset
#                   ,extended=True)               # Printing extra stuff 

# scores =test_model(model = model,test_data = data_fusion(CBC[7],NOISE[7]),extended=True)     #          




                
                   
