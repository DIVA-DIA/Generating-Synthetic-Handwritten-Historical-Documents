
classes = '_!"#*&\'()[]+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

data_name = 'word'
cnn_cfg = [(2, 32), 'M', (4, 64), 'M', (6, 128), 'M', (2, 256)]
rnn_cfg = (256, 1)  # (hidden , num_layers)

max_epochs = 25
batch_size = 32
iter_size = 16
# fixed_size

model_path = '/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/saved_models/'
save_model_name = 'IAM2.pt' #'bla.pt' #'crnn_' + data_name + '_only_iam.pt'
load_model_name = None
#load_model_name = 'GEN-LC.pt' #'Only_gen.pt' #'crnn_' + data_name + '_only_iam.pt'#'_lowercase_new_gen.pt'#'crnn_' + data_name + '_only_gen.pt'
