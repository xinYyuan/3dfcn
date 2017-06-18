from os import listdir
from os.path import isfile, join
import argparse
import h5py
import math

import numpy as np
from scipy import misc
from sklearn.metrics import  mean_squared_error
import network
import network_decov
import  network2d
import network3d
import load_data
import network_test
import matplotlib.pyplot as plt
#import evaluate
import scipy.io as sio
from keras import backend as K

input_size = 33
label_size = 21
pad = (33 - 21) // 2


class SRCNN(object):
    def __init__(self, weight):
        self.model = network3d.srcnn2((None, None,None,1))#(None, None, 1))
        self.model.summary()
        #self.model = network2d.srcnn((None, None, 1))
        f = h5py.File(weight, mode='r')
        self.model.load_weights_from_hdf5_group(f['model_weights'])

    def predict(self, data, **kwargs):
        use_3d_input = kwargs.pop('use_3d_input', True)
        if use_3d_input :
            #if data.ndim != 3:
                #raise ValueError('the dimension of data must be 3 !!')
            #channels = data.shape[0]
            #im_out = [self.model.predict(data[i,None, :, :, :, :]) for i in range(channels)]
            im_out = [self.model.predict(data)]
            #get_3rd_layer_output = K.function([self.model.layers[0].input],
            #                                  [self.model.layers[0].output])
            #im_out = get_3rd_layer_output([data])
        else:
            im_out = [self.model.predict(data)]
            if data.ndim != 2:
                raise ValueError('the dimension of data must be 2 !!')
            im_out = self.model.predict(data[None, :, :, None])
        return np.asarray(im_out)

def show_picture(data):
    plt.imshow(data,plt.cm.gray)
    plt.show()

def test_for_all_bands(input,label):
    #label = label[:, :, 4:-4]
    #label = label[:, :, 4:-4]
    #input=np.reshape(input,[1,input.shape[0],input.shape[1],input.shape[2],1])
    #label=np.reshape(label,[1,label.shape[0],label.shape[1],label.shape[2],1])
    input_new=np.zeros([1,input.shape[0],input.shape[1],input.shape[2],1])
    label_new=np.zeros([1,label.shape[0],label.shape[1],label.shape[2],1])

    for i in range(input.shape[2] ):
        # print i
        input_new[0, :, :, i,0] = input[:, :, i]
        label_new[0, :, :, i,0] = label[:, :, i]

    #print label.shape
    #print  input.shape
    return input_new, label_new[:,:,:,4:-4,:]

def predict():

    srcnn = SRCNN(option.model)
    #srcnn.summary()



    #f= sio.loadmat('/home/yx/HSSR_2.0/data/cuprite/cu_test(mirror9)_(2).mat')
    #f = sio.loadmat('/home/yx/HSSR_2.0/data/pavia/pavia(mirror9)_4_bic.mat') #pa_test(mirror9)_111_snr30.mat')
    #f= sio.loadmat('/home/yx/HSSR_2.0/data/paviau/pau_test(mirror9)_(2)_snr60.mat')
    #f = sio.loadmat('/home/yx/HSSR_2.0/data/salina/sa_test(mirror9)_232.mat')
    #f = sio.loadmat( '/home/yx/HSSR_2.0/data/Indian_pines/indian_test(mirror9).mat')
    #f = sio.loadmat('/home/yx/HSSR_2.0/data/pavia/pa_test(mirror9)_111_snr60.mat')
    f = sio.loadmat('/home/yx/HSSR_2.0/data/urban/urban_mirror9_bic.mat')
    #f = sio.loadmat('/home/yx/HSSR_2.0/data/DC/dc_test(mirror9)_bic.mat')
    input=f['dataa']
    label=f['label']
    print  label.shape
    #input2, label2 = test_for_one_bands(input, label)
    #print ('label2',label2.shape)
    psnr(label[ 6:-6, 6:-6, 4:-4], input[6:-6, 6:-6, 4:-4])
    ssim(label[ 6:-6, 6:-6, 4:-4], input[6:-6, 6:-6, 4:-4])
    sam(label[ 6:-6, 6:-6, 4:-4], input[6:-6, 6:-6, 4:-4])

    input,label=test_for_all_bands(input,label)
    #input=np.reshape(input,[1,input.shape[0],input.shape[1],input.shape[2],1])
    #label=np.reshape(label,[1,label.shape[0],label.shape[1],label.shape[2],1])
    print  input.shape

    output = srcnn.predict(input[:,:,:,:,:])
    print label.shape
    print output.shape

    #file= h5py.File('predict.h5','w')
    #file.create_dataset("output",data=output[0,0,:, :, :,0])
    #file.create_dataset('label',data=label[0,:,:,4:-4,0])
    #print ('save')

    #show_picture(output[0,0,:,:,98,0])
    show_picture(output[0,0,:, :, 25,0 ])
    show_picture(input[0,  :, :, 25, 0])
    ##show_picture(input[98, :, :, 4, 0])
    show_picture(label[0,  :, :, 25, 0])
    print '123'
    '''
    file = h5py.File('predict.h5', 'r')

    label=file['label'][6:-6,6:-6,:]
    output=file['output']

    label=np.float32(label)
    output=np.float32(output)

    #psnr(label,output)
    sam(label,output)
    '''

    psnr(label[0,6:-6,6:-6,:,0],output[0,0,:,:,:,0])
    #psnr(label[:,3:-3,3:-3],output[0,:,:,:,0,0])
    ssim(label[0,6:-6,6:-6,:,0],output[0,0,:,:,:,0])
    sam(label[0,6:-6,6:-6,:,0],output[0,0,:,:,:,0])


def psnr(x_true, x_pred):
    #n_samples = x_true.shape[0]
    print x_true.shape
    print x_pred.shape
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true=x_true[:,:,:]
    for k in range(n_bands):
        x_true_k = x_true[  :, :,k].reshape([-1])
        x_pred_k = x_pred[  :, :,k,].reshape([-1])

        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )

        # print (MSE[k])
        MAX_k = np.max(x_true_k)
        if MAX_k != 0 :
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
            #print ('P', PSNR[k])
        else:
            mask[k] = 0
    # print ('m',mask.sum())
    #psnr= PSNR.mean()
    psnr = PSNR.sum() / mask.sum()
    mse = MSE.mean()
    print('psnr', psnr)
    print('mse', mse)
    return psnr, mse

def ssim(x_true,x_pre):
    #num = (x_true.shape[0])*x_true.shape[1]
    num=x_true.shape[2]
    ssimm=np.zeros(num)
    c1=0.0001
    c2=0.0009
    n=0
    for x in range(x_true.shape[2]):

            #z=np.reshape(x_pre[:,0,0,x,y],[-1])
            #print x_pre.shape
            z = np.reshape(x_pre[:, :,x], [-1])
            sa=np.reshape(x_true[:,:,x],[-1])
            y=[z,sa]
            cov=np.cov(y)
            oz=cov[0,0]
            osa=cov[1,1]
            ozsa=cov[0,1]
            ez=np.mean(z)
            esa=np.mean(sa)
            ssimm[n]=((2*ez*esa+c1)*(2*ozsa+c2))/((ez*ez+esa*esa+c1)*(oz+osa+c2))
            n=n+1
    SSIM=np.mean(ssimm)
    print ('SSIM',SSIM)

def sam(x_true,x_pre):
    print x_pre.shape
    print x_true.shape
    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[ x, y,:], [-1])
            sa = np.reshape(x_true[x, y,:], [-1])
            tem1=np.dot(z,sa)
            tem2=(np.linalg.norm(z))*(np.linalg.norm(sa))
            samm[n]=np.arccos(tem1/tem2)
            n=n+1
    SAM=(np.mean(samm))*180/np.pi
    print('SAM',SAM)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--model',
                        #default='./save_all/save_4/model_pau_150.h5',
                        #default='./save_all/save5/model_pau_9__30.h5',#model_pau_10.h5',
                        #default='./save_all/save_6/model_pau_9_snr60_80.h5',
                        #default='./save_all/save_factor/model_pa_4_330.h5',
                        #default='./save_all/save8/model_pa_8_340.h5',
                        #default='./save_all/save2/model_pa_snr60_130.h5',
                        #default='./save_all/save-liuqi/model_pa_snr60_150.h5',
                        #default='./save_all/save_fine/model_pa_150_150.h5',
                        #default='./save_all/save_fine/model_pau_150_150.h5',
                        #default='./save_all/save_4/model_sa_100.h5',
                        #default='./save_all/save/model_cu_50.h5',
                        #default='./save_all/save_4/model_pau_130.h5',
                        #default='./save_all/save1/model_pa_130.h5',
                        #default='./save_all/save_4/model_cu_50.h5',
                        default='./save_all/save59/model_dc(2)_350.h5',
                        #default='./save_all/save59/model_dc_200.h5',
                        #default='./save_all/save59/model_dc(2)_220.h5',
                        #default='./save_all/save59(2)/model_dc_fine100_300.h5',
                        dest='model',
                        type=str,
                        nargs=1,
                        help="The model to be used for prediction")
    parser.add_argument('-I', '--input-file',
                        default='./dataset/Test/Set5/baby_GT.bmp',
                        dest='input',
                        type=str,
                        nargs=1,
                        help="Input image file path")
    parser.add_argument('-O', '--output-file',
                        default='./dataset/Test/Set5/baby_SRCNN.bmp',
                        dest='output',
                        type=str,
                        nargs=1,
                        help="Output image file path")
    parser.add_argument('-B', '--baseline',
                        default='./dataset/Test/Set5/baby_bicubic.bmp',
                        dest='baseline',
                        type=str,
                        nargs=1,
                        help="Baseline bicubic interpolated image file path")
    parser.add_argument('-S', '--scale-factor',
                        default=2.0,
                        dest='scale',
                        type=float,
                        nargs=1,
                        help="Scale factor")
    option = parser.parse_args()


    predict()