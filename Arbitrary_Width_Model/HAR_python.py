import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
import os
import argparse
from tensorflow.keras import layers
from tensorflow.data import Dataset
from preprocess.UCI import UCI_preprocess as proc
from preprocess.oppo.preprocess_data import *
import matplotlib.pyplot as plt
from preprocess.wisdm import wisdm_preprocess as wis
from preprocess.oppo import slide_oppo_with_null as sop
def choose_parameters():
    msg="Training configuration"
    conf= argparse.ArgumentParser(description=msg)
    conf.add_argument("-dataset", type=str, default="UCI")
    conf.add_argument("-clas", type=int, default=6)
    conf.add_argument("-batch", type=int, default=128)
    conf.add_argument("-learn", type=float, default=5e-4)
    conf.add_argument("-epoch", type=int, default=5)  
    conf.add_argument("-opti", type=str, default="adam")
    conf.add_argument("-width", type=float, default="0.25")
    conf.add_argument("-mtype",type=str,default="awn")
    conf.add_argument("-modtype",type=str,default="test")
    args = conf.parse_args()
    return args

def print_conf():
    print("dataset used:{}".format(args.dataset))
    print("class number:{}".format(args.clas))
    print("batch size:{}".format(args.batch))
    print("learning rate:{}".format(args.learn))
    print("training epoch:{}".format(args.epoch))
    print("optimizer used:{}".format(args.opti))
    print("Width used:{}".format(args.width))
    
def label_dataset(dname):
    if dname == 'UCI':
        labels = ['Sitting', 'Standing', 'Laying','Walking', 'Walking Upstairs', 'Walking Downstairs' ]
    if dname == 'wisdm':
        labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
    if dname=='oppo':
        labels = ['Null', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2', 'Open Fridge',
                           'Close Fridge', 'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1',
                           'Open Drawer 2', 'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3',
                           'Clean Table', 'Drink from Cup', 'Toggle Switch']
    if dname=='unimib':
        labels =  ['StandingUpFS', 'StandingUpFL', 'Walking', 'Runing', 'GoingUpS', 'Jumping', 'GoingDownS',
                           'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack',
                           'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft']
    if dname=='pamap2':
        labels = ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
                           'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping']
    return labels
    
def make_whole(x, div=8, value_min=1):
    
    if value_min is None:
       value_min = div
    mod_x = max(value_min, int(x + div / 2) // div * div)
    if mod_x < 0.9 * x:
        mod_x += div
    return mod_x
    
    
class load_data:
    def __init__(self,x_file,y_file):
        self.x_file=x_file
        self.y_file=y_file
    def HAR_data(self):
        x_data_np = np.load(self.x_file)
        x_data = x_data_np.reshape(-1, x_data_np.shape[1], x_data_np.shape[2],1)
        y_data = np.load(self.y_file)
        tensor_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        return tensor_dataset

class SlimNN(tf.keras.layers.Layer):
    
    #def __init__(self, width, fixed_chan=False, div=1, min_chan=1):
    def slimnn_init(self, width, fixed_chan=False, div=1, min_chan=1):
        self.set_width(width)
        self.set_fixed_channels(fixed_chan)
        self.set_divisor(div)
        self.set_min_chan(min_chan)
    def set_divisor(self, div=1):
        self.div = div
    def set_width(self, width):  
        self.check_width(width)
        self.curr_slice = width
    def set_min_chan(self, mini=1):
        self.min_chan = mini
    def set_fixed_channels(self, fixed=True):
        self.fixed_chan = fixed
    def round_slice(self, channels, width=None):
        if width is None:
            width = self.curr_slice
        return make_whole(round(width * channels),self.div, self.min_chan) 
    @staticmethod
    def check_width(width):
        if not width> 0.0 and width <= 1.0:
            raise Exception("Invalid width value")

class SlimmableConv2d(tf.keras.layers.Conv2D, SlimNN):

    def __init__(self,filters,kernel_size,strides=1,padding="valid", dilation_rate=1,groups=1,use_bias=True,fixed_chan=False, div=1, min_chan=1):

        super(SlimmableConv2d, self).__init__(filters,kernel_size)
       
        self.slimnn_init(1.0, fixed_chan, div, min_chan)
        
        y = tf.keras.layers.Conv2D(filters,kernel_size, activation='relu',kernel_initializer="glorot_uniform")
        print(y.get_weights())
        
        #print(weight[0].shape)
        
        
    def call(self, input):
        #x = tf.random.normal(input)
        #print(x)
        l = tf.keras.layers.Conv2D(input_shape=input.shape[1:], filters=1, kernel_size=3, padding='valid')
        z=l(input)
        #print(l.trainable_variables)
        weights = l.get_weights()
        """
        var = [v for v in l.trainable_variables if v.name == "conv2d/kernel:0"][0]
        print(var)
        print(len(weights))
        for i in range(len(weights)):
            print(weights[i].shape)
        """
        
        weight=weights[0]
        weight_shape=weights[0].shape
        bias=weights[1]
        bias_shape=weights[1].shape
        print(weight_shape)
        
        out_channels,in_channels = weights[0].shape[2:]
        if self.fixed_chan:
            sliced_out_channels = out_channels
        else:
            sliced_out_channels = self.round_slice(out_channels)
        #sliced_out_channels = (out_channels if self.fixed_chan else self._slice(out_channels))
        sliced_in_channels = input.shape[3]
        sliced_weight= np.array(weight[ :, :, :sliced_out_channels, :sliced_in_channels])
        if self.use_bias is False:
            sliced_bias = bias
        else:
            sliced_bias = np.array(bias[:sliced_out_channels])
        #sliced_bias = (None if bias is None else bias[:sliced_out_channels]))
        print(sliced_weight,sliced_bias)
        print(sliced_weight.shape,sliced_bias.shape)
        out = tf.keras.layers.Conv2D(input_shape=input.shape[1:], filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding ,dilation_rate=self.dilation_rate,groups=self.groups)
        res=out(input)
        print(len(out.get_weights()))
        weight_parameter=[sliced_weight,sliced_bias]
        out.set_weights(weight_parameter)
        #out.set_weights(sliced_weight,sliced_bias)  
        
        #out.set_weights(np.array([[weight_shape[0], weight_shape[1],sliced_out_channels, sliced_in_channels],[sliced_bias]]))     
	    #print(len(out.get_weights()))
        #out = tf.nn.conv2d(input, sliced_weight, self.stride, self.padding ,self.dilation_rate,self.groups)
        return out

class SlimmableDense(tf.keras.layers.Dense, SlimNN):
    def __init__(self,units,activation=None,use_bias=True,fixed_chan=False, div=1, min_chan=1):
        super(SlimmableDense, self).__init__(units,activation, use_bias)
        self.slimnn_init(1.0, fixed_chan, div, min_chan)
    
    def call(self, input):
        
        #x = tf.random.normal(input)
        dlayer=tf.keras.layers.Dense(self.units,self.activation,self.use_bias)
        a=dlayer(input)
        weight=dlayer.get_weights()
        dweight=weights[0]
        weight_shape=weight[0].shape
        bias=weight[1]
        bias_shape=weight[1].shape
        print(weight_shape)
        out_channels, in_channels = weight[0].shape
        if self.fixed_chan:
            sliced_in_channels = in_channels
        else:
            sliced_in_channels = input.shape[3]
        #sliced_in_channels = (in_channels if self.fixed_channels else input.shape[3])
        sliced_out_channels = (out_channels if self.fixed_chan else self._slice(out_channels))

        sliced_weight = np.array(dweight[:sliced_out_channels, :sliced_in_channels])
        if self.use_bias==False:
            sliced_bias = bias
        else:
            sliced_bias = bias[:sliced_out_channels]
        #sliced_bias = (None if self.use_bias is False else dlayer.get_weights[1][:sliced_out_channels])


        out = tf.keras.layers.Dense(self.units,self.activation)
        x=out(input)
        weight_parameter=[sliced_weight,sliced_bias]
        out.set_weights(weight_parameter)
        return out
        
class BatchNorm2dSlimmable(tf.keras.layers.BatchNormalization, SlimNN):
  
    def __init__(self, momentum=0.99,epsilon= 0.00001, scale= True, center=True, training = True,fixed_chan=False, div=1, min_chan=1):
        super(BatchNorm2dSlimmable, self).__init__(momentum, epsilon)
        self.slimnn_init(1.0, fixed_chan, divisor, min_channels)

    def call(self, input):
        
        bn = tf.keras.layers.BatchNormalization(input,self.momentum, self.epsilon, self.training)
        y=bn(input)
        no_batch_tracked= bn.num_batches_tracked
        moving_mean = bn.moving_mean
        moving_var=bn.moving_variance
        average_normalized_value = 0.0
        if self.training is True:
            if no_batch_tracked is not None:
                no_batch_tracked.assign_add(1)
                print(no_batch_tracked.numpy())
                if self.momentum is None:  
                    average_normalized_value = 1.0
                else:  
                    average_normalized_value = self.momentum
        batchnorm_weights=bn.get_weights()
        b_weight=batchnorm_weights[0]
        b_bias=batchnorm_weights[1]
        sliced_channels = self.round_slice(len(b_weight))
        if not abs(sliced_channels - input.shape[3] <= self.divisor):
            raise Exception("Expected {} but got {}".format(sliced_channels,input.shape[3]))
        sliced_channels = input.shape[3]

        sliced_weight = sliced_bias = None
        sliced_running_mean = sliced_running_var = None
        if self.scale is True and self.center is True:
            sliced_weight = b_weight[:sliced_channels]
            sliced_bias = b_bias[:sliced_channels]
        if self.training:
            sliced_running_mean = moving_mean[:sliced_channels]
            sliced_running_var = moving_var[:sliced_channels]

        out = tf.nn.batch_normalization(input, sliced_running_mean, sliced_running_var,self.training,average_normalized_value, self.eps)
        
        weight_parameter=[sliced_weight,sliced_bias]
        out.set_weights(weight_parameter)
        
        return out

class SharedSlimmableBatchNorm2d(SlimNN):

    def __init__(self,inp_val, width, momentum=0.99,epsilon= 0.00001,scale=True,center=True, training=True,
                 fixed_chan=False, div=1, min_chan=1):
        super(SharedSlimmableBatchNorm2d, self).__init__()
        
        self.set_divisor(div)
        self.set_min_chan(min_chan)
        self.set_fixed_channels(fixed_chan)
        self.widths = width
        self.bn_dict = tf.ModuleDict()  
        for width in self.widths:
            self.bn_dict["width_"+self.rep_point(width)] = BatchNorm2dSlimmable(
                self.round_slice(inp_val, width), self.momentum,self.epsilon, selsf.scale,self.center,
                self.training, self.div, self.min_chan)
            z= self.bn_dict["width_"+self.rep_point(width)](inp_val)
        self.set_width(1.0)

    def rep_point(self, float_val):
        return str(float_val).replace('.', '_')

    def to_bn_key(self, width):
        return min([s for s in self.widths if s >= width])

    def set_width(self, width):  
        self.check_width(width)
        self.curr_width = width
        self.curr_bn_key = self.to_bn_key(self.curr_slice)
        self.curr_bn = self.bn_dict[self.to_mdict_key(self.curr_bn_key)]
        self.curr_bn.set_width(self.curr_width / self.curr_bn_key)  


    def call(self, input):
        return self.curr_bn(input) 


class SlimmableTriangularConv2d(SlimmableConv2d):

    def __init__(self, filters,kernel_size,strides=1,padding="valid", dilation_rate=1,groups=1,use_bias=True,
                 fixed_chan= False, div=1, min_chan=1,min_width=0.000001, sampled_width=1000): 
        super(MaskTriangularConv2d, self).__init__(
            filters, kernel_size, strides, padding, dilation_rate,
            groups, use_bias, fixed_chan, divisor, min_channels)
        
        self.sampled_width_mults = sampled_width
        self.generate_mask(min_width)
        return

    def generate_mask(self, min_width):
        self.check_width(min_width)
        self.min_width = min_width
        self.mask = self.make_mask()
        return

    def reduce_weights(self, width_mult):
        in_chan,out_chan = weight.shape[2:]
        if self.fixed_channels:
            trimmed_in_chan = in_chan
        else:
            trimmed_in_chan = self.round_slice(in_chan,width_mult)
        
        red_out_channels = (out_chan if self.fixed_channels else self.round_slice(out_chan, width_mult))
        return (sliced_in_channels, sliced_out_channels)

    def make_mask(self):
        ht,wth,chan_in,chan_out = weight.shape
        array=np.array([ht,wth,chan_in,chan_out])
        mask = tf.zeros(array)

        max_width = 1.0
        min_width = self.min_width
        chan_out_min = (chan_out if self.fixed_channels
                     else self.round_slice(chan_out, min_width))
        chan_in_min = (chan_in if self.fixed_channels
                    else self.round_slice(chan_in, min_width))

        mask[:, :, :min_C_out, :min_C_in] = 1.0
        final_chan_out = chan_out_min
        max_chan  = max(chan_out, chan_in, self.sampled_width)
        width_mults = sorted([min_w, max_w]
            + list(np.random.uniform(min_w, max_w, max_C)))
        count=0  
        for x in width_mults:
            count+=1  
            new_chan_in, new_chan_out = self.reduce_weights(x)
            mask[:, :, :new_C_in,final_chan_out:final_chan_out] = 1.0
            last_chan_out = new_chan_out
            

        mask= tf.Variable(mask,trainable=False)
        self.num_active_weights = mask.numpy().item()

        return mask

    def call(self, input):
    
        layer = tf.keras.layers.Conv2D(input.shape[1:],self.filters,self.kernel_size,self.strides,self.padding,
                            self.dilation_rate,self.groups,self.use_bias)
        temp=layer(input)
        get_weights=layer.get_weights()
        weight=get_weights[0]
        bias=get_weights[1]
        
        new_weight=tf.multiply(weight,self.mask)
        mod_in_chan,trimmed_out_channels = self.reduce_weights(self.curr_width)
        red_in_channels = input.shape[3]

        trimmed_weight = weight[ :, :, :red_in_channels, :trimmed_out_channels]
        if bias is None:
            red_bias= None
        else:
            red_bias=bias[:sliced_out_channels]
        #red_bias = (None if bias is None else bias[:sliced_out_channels])

      
        out = tf.keras.layers.Conv2D(input.shape[1:],self.strides,self.padding,self.dilation_rate,self.groups)
        temp_layer=out(input)
        modified_wts=[trimmed_weights,red_bias]
        out.set_weights(modified_wts)
        
        return out

class AWNet(SlimNN):
    
    def set_width(self, width):  
        self.check_width(width)
        self.curr_slice = slice

        def _set_width_mult(layer):
            if isinstance(layer, SlimNN) and layer is not self:
                layer.set_width(self.curr_width)
        self.apply(_set_width_mult)

    def call(self, inp):
        inp = tf.keras.Model(inp)
        inp = tf.reshape(inp,(inp.size(0), -1))
        inp = tf.keras.layers.Dense(inp)
        return inp

def to_one_hot(y, no_of_dims=None):
    
    y = tf.cast(y, dtype=tf.int64)
    y_value = tf.reshape(y, [-1, 1])
    no_of_dims = no_of_dims if no_of_dims is not None else tf.cast(tf.reduce_max(y) + 1, tf.int32)
    zeros_tensor = tf.zeros([y_value.shape[0], no_of_dims])
    updates = tf.ones([y_value])
    indices = tf.stack([tf.range(batch_size), y_value], axis=1)
    ones_tensor = tf.tensor_scatter_nd_update(zeros_tensor, indices, updates)
    y_one_hot = tf.reshape(y_one_hot, [*y.shape, -1])
    return y_one_hot

def adjust_learn_rate(optim, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50))  
    for var in optimizer.variables():
        if 'learning_rate' in var.name:
            var.assign(lr)
        
       
def train(model, train_batch, loss_fn, optim, learn_rate, epoch):
    adjust_learn_rate(optim, epoch, learn_rate)
    accuracy = tf.keras.metrics.Accuracy()
    model.trainable=True
    train_loss_metric = tf.keras.metrics.Mean()
    loss_epoch = 0
    acc_epoch = 0
    loss_list = []

    for step, (x, y) in enumerate(train_batch):
        train_loss_metric.reset_states()
        #inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
        batch_x = tf.cast(batch_x, tf.float32)
        with tf.device('/GPU:0'):
            inputs = tf.Variable(batch_x)
        batch_y = tf.cast(batch_y, tf.int64)
        with tf.device('/GPU:0'):
            labels = tf.Variable(batch_y)

        labels_conv_onehot = to_one_hot(labels)
        with tf.GradientTape() as tape:
            
            # Forward pass
            y_pred = model(inputs)
            preds = tf.argmax(outputs, axis=1)
            accuracy.update_state(labels, preds)
            epoch_acc = accuracy.result().numpy()
            # Compute loss
            loss = loss_fn(y_pred,labels)

        # Compute gradients
        grads = tape.gradient(loss, model.trainable_variables)
        # Update model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)
        train_loss = train_loss / len(train_batch)
        # Clear gradients
        tape.reset()
        

        print("Training epoch {}, Loss {}, Accuracy {}, Learning Rate {}".format(step, loss.numpy(),accuracy,optim.learning_rate.numpy()))

    return train_loss, epoch_acc



def test(model, test_batch, loss_fn):  
    test_loss_metric = tf.keras.metrics.Mean()
    epoch_loss= 0
    epoch_acc = 0
    accuracy = tf.keras.metrics.Accuracy()
    conf_matrix = np.array(0)
    num_batches=0
    model.eval()
    
    for step, (x,y) in enumerate(test_batch):
        batch_x = tf.cast(batch_x, tf.float32)
        with tf.device('/GPU:0'):
            inputs = tf.Variable(batch_x)
        batch_y = tf.cast(batch_y, tf.int64)
        with tf.device('/GPU:0'):
            labels = tf.Variable(batch_y)
        with tf.GradientTape() as tape:
        #inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            y_pred = model(inputs)
            preds = tf.argmax(outputs, axis=1)
            accuracy.update_state(labels, preds)
            loss = loss_fn(y_pred,labels)
            #batch_loss = compute_loss(batch_x, batch_y)
            epoch_loss += loss.numpy()
            num_batches += 1
        grads = tape.gradient(loss, model.trainable_variables)
        # Update model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        test_loss_metric.update_state(loss)
        test_loss = test_loss / len(test_batch)
        # Clear gradients
        tape.reset()

    
    return epoch_loss, epoch_acc, conf_matrix
    
class AdaptiveCNN_uci(AWNet):
    def __init__(self, no_of_classes=6,init_width=1, slices=[1.0], div=1, min_chan=1):
        super(AdaptiveCNN_uci, self).__init__()

        self.set_width(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_chan)

        n = self.round_slice(128, init_width)
        in_chan = 1
        

        log_slices = [0.25, 0.5, 0.75, 1.0]
        self.features = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(n, (6, 1), (3, 1), fixed_chan=True),
            SharedSlimmableBatchNorm2d(n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(2*n, (6, 1), (3, 1)),
            SharedSlimmableBatchNorm2d(2*n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'),
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(4*n, (6, 1), (3, 1)),
            SharedSlimmableBatchNorm2d(4*n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid')
            ])
        self.flatten = layers.Flatten()
       

        self.classifier = keras.Sequential([
            SlimmableDense(num_classes, fixed_chan=True) 
        ])

    def call(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
 
class AdaptiveCNN_oppo(AWNet):
    def __init__(self, no_of_classes=18,init_width=1.0, slices=[1.0], div=1, min_chan=1):
        super(AdaptiveCNN_oppo, self).__init__()

        self.set_width_mult(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        n = self._slice(128, init_width_mult)
        inC = 1
        log_slices = [0.25, 0.5, 0.75, 1.0]
        self.layer1 = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,1)),
            SlimmableTriangularConv2d(n, 3, 2),
            SharedSlimmableBatchNorm2d(n, slices),
            keras.layers.ReLU(),
            ])
        self.layer2 = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,1)),
            SlimmableTriangularConv2d(n, 2*n, 3, 2),
            SharedSlimmableBatchNorm2d(2*n, slices),
            keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            keras.layers.MaxPool2d((2, 1), (1, 1)),
            ])
        self.layer3 = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,1)),
            SlimmableTriangularConv2d(4*n, 3, 2),
            SharedSlimmableBatchNorm2d(4*n, slices),
            keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            keras.layers.MaxPool2d((2, 1), (1, 1)),
        ])
        self.flatten = layers.Flatten()

        self.classifier = keras.Sequential([
            SlimmableDense( no_of_classes, fixed_chan=True)  
        ])

    def call(self, inp):
        inp = self.layer1(inp)
        inp = self.layer2(inp)
        inp = self.layer3(inp)
        inp = self.flatten(inp)
        inp = self.classifier(inp)
        return x 

class AdaptiveCNN_pamap2(AWNet):
    def __init__(self, no_of_classes=12,init_width=1, slices=[1.0], div=1, min_chan=1):
        super(AdaptiveCNN_pamap2, self).__init__()

        self.set_width(1.0)
        self.set_divisor(div)
        self.set_min_channels(min_chan)

        n = self.round_slice(128, init_width)
        in_chan = 1
        

        log_slices = [0.25, 0.5, 0.75, 1.0]
        self.features = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(n, (6, 1), (3, 1), fixed_chan=True),
            SharedSlimmableBatchNorm2d(n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(2*n, (6, 1), (3, 1)),
            SharedSlimmableBatchNorm2d(2*n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'),
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(4*n, (6, 1), (3, 1)),
            SharedSlimmableBatchNorm2d(4*n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid')
        ])
        self.flatten = layers.Flatten()
       
        self.classifier = keras.Sequential([
            SlimmableDense(no_of_classes, fixed_chan=True) 
        ])

    def call(self, inp):
        inp = self.features(inp)
        inp = self.flatten(inp)
        inp = self.classifier(inp)
        return x        

class AdaptiveCNN_unimib(AWNet):
    def __init__(self, num_classes=17,init_width=1.0, slices=[1.0], div=1, min_chan=1):
        super(AdaptiveCNN_unimib, self).__init__()

        self.set_width_mult(1.0)
        self.set_divisor(div)
        self.set_min_channels(min_chan)

        n = self._slice(128, init_width)
        in_chan = 1
    

        log_slices = [0.25, 0.5, 0.75, 1.0]
        self.features = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(n, (6, 1), (2, 1), fixed_chan=True),
            SharedSlimmableBatchNorm2d(n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(2*n, (6, 1), (2, 1)),
            SharedSlimmableBatchNorm2d(2*n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'),
            layers.ZeroPadding2D(padding=(1,0)),
            SlimmableTriangularConv2d(4*n, (6, 1), (2, 1)),
            SharedSlimmableBatchNorm2d(4*n, slices),
            tf.keras.layers.ReLU(),
            layers.ZeroPadding2D(padding=(1,0)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid')
        ])
        self.flatten = FlattenLayer()

        self.classifier = keras.Sequential([
            SlimmableDense(no_of_classes, fixed_chan=True)  
        ])

    def call(self, inp):
        inp = self.features(inp)
        inp = self.flatten(inp)
        inp = self.classifier(inp)
        return inp

class AdaptiveCNN_wisdm(AWNet):
    def __init__(self, no_of_classes=6,init_width=1.0, slices=[1.0], div=1, min_chan=1):
        super(AWLeNet5_wisdm, self).__init__()

        self.set_width_mult(1.0)
        self.set_divisor(div)
        self.set_min_channels(min_chan)

        n = self.round_slice(128, init_width)
        inC = 1
        

        log_slices = [0.25, 0.5, 0.75, 1.0]
        
        self.features = keras.Sequential([
            layers.ZeroPadding2D(padding=(1,1)),
            SlimmableTriangularConv2d(n, 3, 2, fixed_chan=True),
            SharedSlimmableBatchNorm2d(n, slices),
            layers.ReLU(inplace=True),
            layers.ZeroPadding2D(padding=(1,1)),
            SlimmableTriangularConv2d(2 * n, 3, 2),
            SharedSlimmableBatchNorm2d(2 * n, slices),
            layers.ReLU(inplace=True),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'),
            layers.ZeroPadding2D(padding=(1,1)),
            SlimmableTriangularConv2d(4 * n, 3, 2),
            SharedSlimmableBatchNorm2d(4 * n, slices),
            layers.ReLU(inplace=True),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'),
        ])

        self.flatten = layers.Flatten()

        self.classifier = keras.Sequential([
            SlimmableDense(no_of_classes, fixed_chan=True)  
        ])

    def call(self, inp):
        inp = self.features(inp)
        inp = self.flatten(inp)
        inp = self.classifier(inp)
        return x
        
def uci_cnn(x_train,y_train,x_test,y_test):
            
            channels=[128, 256, 512, 23040]
            model= keras.Sequential()
            model.add(Input(shape=(128,9,1)))
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[0], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[1], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[2], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(6,activation='softmax'))
            model.summary()
            #return model
            #batch_size=128
            #epochs=5
            
           
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics=["accuracy"])
            model.fit(x_train,y_train,batch_size=args.batch,epochs=args.epoch,validation_split=0.1)
            scores = model.evaluate(x_test, y_test, verbose=0) 
            print("Accuracy: %.2f%%" % (scores[1]*100))
            
def wisdm_cnn(x_train,y_train,x_test,y_test):
            channels=[128, 256, 512, 23040]
            
            model= keras.Sequential()
            model.add(Input(shape=(200,3,1)))
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[0], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[1], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[2], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(6,activation='softmax'))
            model.summary()
            #return model
            batch_size=128
            #epochs=25
            
            
            #x_train = x_train.reshape(-1, 1, 128, 9)
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics=["accuracy"])
            model.fit(x_train,y_train,batch_size=batch_size,epochs=args.epoch,validation_split=0.1)
            scores = model.evaluate(x_test, y_test, verbose=0) 
            print("Accuracy: %.2f%%" % (scores[1]*100))
            
def pamap2_cnn(x_train,y_train,x_test,y_test):
            
            channels=[128, 256, 512, 23040]
            model= keras.Sequential()
            model.add(Input(shape=(171,40,1)))
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[0], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[1], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[2], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(6,activation='softmax'))
            model.summary()
            #return model
            #batch_size=128
            #epochs=5
            
           
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics=["accuracy"])
            model.fit(x_train,y_train,batch_size=args.batch,epochs=args.epoch,validation_split=0.1)
            scores = model.evaluate(x_test, y_test, verbose=0) 
            print("Accuracy: %.2f%%" % (scores[1]*100))
 
def oppo_cnn(x_train,y_train,x_test,y_test):
            channels=[128, 256, 512, 23040]
            
            model= keras.Sequential()
            model.add(Input(shape=(200,3,1)))
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[0], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[1], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[2], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(6,activation='softmax'))
            model.summary()
            #return model
            batch_size=128
            #epochs=25
            
            
            #x_train = x_train.reshape(-1, 1, 128, 9)
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics=["accuracy"])
            model.fit(x_train,y_train,batch_size=batch_size,epochs=args.epoch,validation_split=0.1)
            scores = model.evaluate(x_test, y_test, verbose=0) 
            print("Accuracy: %.2f%%" % (scores[1]*100))

def unimib_cnn(x_train,y_train,x_test,y_test):
            
            channels=[128, 256, 512, 23040]
            model= keras.Sequential()
            model.add(Input(shape=(171,40,1)))
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[0], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[1], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.Conv2D(channels[2], (6,1),strides=(3,1), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,0)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='valid'))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(6,activation='softmax'))
            model.summary()
            #return model
            #batch_size=128
            #epochs=5
            
           
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics=["accuracy"])
            model.fit(x_train,y_train,batch_size=args.batch,epochs=args.epoch,validation_split=0.1)
            scores = model.evaluate(x_test, y_test, verbose=0) 
            print("Accuracy: %.2f%%" % (scores[1]*100))
            
def preprocess(dataset_name):

    if dataset_name=="UCI":
        x_train = proc.load_x(proc.x_train_paths_list)
        x_test = proc.load_x(proc.x_test_paths_list)
        print("x_test.shape", x_test.shape)

        y_train = proc.load_y(proc.y_train_path)
        y_test = proc.load_y(proc.y_test_path)
       
        print(y_train, Counter(y_train))
        print(y_test, Counter(y_test))
        np.save("./dataset/UCI/x_train.npy", x_train)
        np.save("./dataset/UCI/y_train.npy", y_train)
        np.save("./dataset/UCI/x_test.npy", x_test)
        np.save("./dataset/UCI/y_test.npy", y_test)
        
    if dataset_name=="wisdm":
    
        segments, labels = wis.segment_signal(wis.dataset)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
        reshaped_segments = segments.reshape(len(segments), 200, 3)  

        train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
        train_x = reshaped_segments[train_test_split]
        train_y = labels[train_test_split]
        test_x = reshaped_segments[~train_test_split]
        test_y = labels[~train_test_split]
        y_train=train_y.flatten()
        y_test=test_y.flatten()

        np.save('./dataset/wisdm/x_train.npy', train_x)
        np.save('./dataset/wisdm/y_train.npy', y_train)
        np.save('./dataset/wisdm/x_test.npy', test_x)
        np.save('./dataset/wisdm/y_test.npy', y_test)
    
    if dataset_name=="Unimib":
        xtrain=np.load('acc_data.npy',allow_pickle=True)
        xtest=np.load('adl_data.npy',allow_pickle=True)
        ytrain=np.load('acc_labels.npy',allow_pickle=True)
        ytest=np.load('adl_labels.npy',allow_pickle=True)
        
        x_train=xtrain.reshape(11771,151,3)
        print(x_train)
        
        x_test=xtest.reshape(7579,151,3)
        print(x_test)
        
        ytrain_arr=ytrain.flatten()
        print(ytrain_arr.shape)
        
        pd.DataFrame(ytrain_arr).to_csv("file_ytrain.csv",index=None,header=None)
        y_train=np.loadtxt("y_train.csv",delimiter=",",dtype=int)
        print(y_train.shape)
        
        ytest_arr=ytest.flatten()
        print(ytest_arr.shape)
        
        pd.DataFrame(ytest_arr).to_csv("file_ytest.csv",index=None,header=None)
        y_test=np.loadtxt("y_test.csv",delimiter=",",dtype=int)
        print(y_test.shape)
        
        np.save('./dataset/Unimib/x_train.npy',x_train)
        np.save('./dataset/Unimib/x_test.npy',x_test)
        np.save('./dataset/Unimib/y_train.npy',y_train)
        np.save('./dataset/Unimib/y_test.npy',y_test)
        
    if dataset_name=="oppo":
        
        dataset_name, sub, l = get_args()
        dataset = find_data(dataset_name)
        generate_data(dataset_name, dataset, sub, l)
        X_train,y_train = sop.load_data2array('train', sop.window_size, sop.stride) 
        print(y_train.shape)
        X_valid, y_valid = sop.load_data2array('val', sop.window_size, sop.stride) 
        print(y_valid.shape)
        X_test, y_test = sop.load_data2array('test', sop.window_size, sop.stride) 
        print(y_test.shape)

        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)
        print(X_test.shape)
        print(y_test.shape)

        np.save("./dataset/oppo/x_train.npy", X_train)
        np.save("./dataset/oppo/y_train.npy", y_train)
        np.save("./dataset/oppo/x_valid.npy", X_valid)
        np.save("./dataset/oppo/y_valid.npy", y_valid)
        np.save("./dataset/oppo/x_test.npy", X_test)
        np.save("./dataset/oppo/y_test.npy", y_test)

def gui():
    test_x_list = "./dataset/UCI/x_test.npy"
    test_y_list = "./dataset/UCI/y_test.npy"

    top = tk.Tk()

    top.title("HAR_demo")
    width = 260
    height = 420
    top.geometry(f'{width}x{height}')

    var_Walking = tk.StringVar()
    var_Upstair = tk.StringVar()
    var_Downstairs = tk.StringVar()
    var_Sitting = tk.StringVar()
    var_Standing = tk.StringVar()
    var_Laying = tk.StringVar()

    var_Walking.set("?")
    var_Upstair.set("?")
    var_Downstairs.set("?")
    var_Sitting.set("?")
    var_Standing.set("?")
    var_Laying.set("?")

    text_Walking = tk.Label(top, text="Walking").place(x=50, y=0)
    text_Upstairs = tk.Label(top, text="Walking Upstairs").place(x=50, y=20)
    text_Downstairs = tk.Label(top, text="Walking Downstairs").place(x=50, y=40)
    text_Sitting = tk.Label(top, text="Sitting").place(x=50, y=60)
    text_Standing = tk.Label(top, text="Standing").place(x=50, y=80)
    text_Laying = tk.Label(top, text="Laying").place(x=50, y=100)  # Jogging

    # time_ = tk.Label(top, text="inference time:").place(x=50, y=140)
    # time_2 = tk.Label(top, text="295.213ms").place(x=170, y=140)

    text_Walking_value = tk.Label(top, textvariable=var_Walking).place(x=200, y=0)
    text_Upstairs_value = tk.Label(top, textvariable=var_Upstair).place(x=200, y=20)
    text_Downstairs_value = tk.Label(top, textvariable=var_Downstairs).place(x=200, y=40)
    text_Sitting_value = tk.Label(top, textvariable=var_Sitting).place(x=200, y=60)
    text_Standing_value = tk.Label(top, textvariable=var_Standing).place(x=200, y=80)
    text_Laying_value = tk.Label(top, textvariable=var_Laying).place(x=200, y=100)


    img_gif = tk.PhotoImage(file='./动作/问号.gif')
    img_gif0 = tk.PhotoImage(file='./动作/走.gif')
    img_gif1 = tk.PhotoImage(file='./动作/上楼.gif')
    img_gif2 = tk.PhotoImage(file='./动作/下楼.gif')
    img_gif3 = tk.PhotoImage(file='./动作/坐.gif')
    img_gif4 = tk.PhotoImage(file='./动作/站立.gif')
    img_gif5 = tk.PhotoImage(file='./动作/躺.gif')

    label_img = tk.Label(top, image=img_gif)
    label_img.place(x=30, y=160)  # 30  120

    def Clear_result():
        var_Walking.set("?")
        var_Upstair.set("?")
        var_Downstairs.set("?")
        var_Sitting.set("?")
        var_Standing.set("?")
        var_Laying.set("?")
        label_img.configure(image=img_gif)

    def uci_gui_test():
        global prob
        global result

        data_test = HAR_one_tensor(test_x_list, test_y_list)
        har_test_tensor = data_test.HAR_one_tensor_data()

        test_loader = Data.DataLoader(dataset=har_test_tensor, batch_size=1, shuffle=True, )
        with tf.device('CPU'):
    
            model = tf.keras.models.load_model('./model_save/UCI/net0.965412004069176_199', compile=False)
            preds_prob, preds = test(model, test_loader)
            prob = np.around(preds_prob.numpy(), decimals=2)[0]
            result = preds

        var_Walking.set(prob[0])
        var_Upstair.set(prob[1])
        var_Downstairs.set(prob[2])
        var_Sitting.set(prob[3])
        var_Standing.set(prob[4])
        var_Laying.set(prob[5])
        if result == 0:
            label_img.configure(image=img_gif0)
        elif result == 1:
            label_img.configure(image=img_gif1)
        elif result == 2:
            label_img.configure(image=img_gif2)
        elif result == 3:
            label_img.configure(image=img_gif3)
        elif result == 4:
            label_img.configure(image=img_gif4)
        elif result == 5:
            label_img.configure(image=img_gif5)


    button = tk.Button(top, text='Prediction', command=uci_gui_test)
    button.place(x=60, y=370)  # button.place(x=60, y=330)

    button_Clear = tk.Button(top, text='Clear', command=Clear_result)
    button_Clear.place(x=150, y=370)  # button_Clear.place(x=150, y=330)

    top.mainloop()
        
if __name__=="__main__":
    """
    dataset_name=input("Enter the dataset name \n")
    preprocess(dataset_name)
    """
    train_acc_list,train_loss_list,test_loss_list,test_acc_list=([] for i in range(4))
    args=choose_parameters()
    print_conf()
    x_train_file='./dataset/'+args.dataset+'/x_train.npy'
    x_test_file='./dataset/'+args.dataset+'/x_test.npy'
    y_train_file='./dataset/'+args.dataset+'/y_train.npy'
    y_test_file='./dataset/'+args.dataset+'/y_test.npy'
    print(x_train_file,y_train_file,x_test_file,y_test_file)
    
    data_labels=label_dataset(args.dataset)
    print(data_labels)
    
    x_train=np.load('./dataset/'+args.dataset+'/x_train.npy')
    x_test=np.load('./dataset/'+args.dataset+'/x_test.npy')
    y_train=np.load('./dataset/'+args.dataset+'/y_train.npy')
    y_test=np.load('./dataset/'+args.dataset+'/y_test.npy')
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    train_data = load_data(x_train_file, y_train_file)
    har_train_data = train_data.HAR_data()
    print(har_train_data)
    
    test_data = load_data(x_test_file, y_test_file)
    har_test_data = test_data.HAR_data()
    print(har_test_data)
    
    train_loader = har_train_data.shuffle(24).batch(args.batch)
    test_loader = har_test_data.shuffle(24).batch(args.batch)
    print(train_loader,test_loader)
    
   
    if args.dataset=="UCI":
        x_train=x_train.reshape(-1,128,9,1)
        x_test=x_test.reshape(-1,128,9,1)
        if args.mtype=="baseline":
            model = uci_cnn(x_train,y_train,x_test,y_test)
        else:
            model = AdaptiveCNN_uci()
            
    if args.dataset=="wisdm":
        x_train=x_train.reshape(-1,200,3,1)
        x_test=x_test.reshape(-1,200,3,1)
        if args.mtype=="baseline":
            model = wisdm_cnn(x_train,y_train,x_test,y_test)
        else:
            model = AdaptiveCNN_wisdm()
            
    if args.dataset=="oppo":
        x_train=x_train.reshape(-1,40,113,1)
        x_test=x_test.reshape(-1,40,113,1)
        if args.mtype=="baseline":
            model = oppo_cnn(x_train,y_train,x_test,y_test)
        else:
            model = AdaptiveCNN_oppo()
            
    if args.dataset=="pamap2":
        x_train=x_train.reshape(-1,171,40,1)
        x_test=x_test.reshape(-1,171,40,1)
        if args.mtype=="baseline":
            model = ConvNet_2d_pamap2(x_train,y_train,x_test,y_test)
        else:
            model = AdaptiveCNN_pamap2()
            
    if args.dataset=="Unimib":
        x_train=x_train.reshape(-1,151,3,1)
        x_test=x_test.reshape(-1,151,3,1)
        if args.mtype=="baseline":
            model = ConvNet_2d_unimib(x_train,y_train,x_test,y_test)
        else:
            model = AdaptiveCNN_unimib()
            
    if args.modtype=="train":
        optim = tf.keras.optimizers.Adam(learning_rate=args.learn)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        acc_best = 0

        for e in range(0, args.epoch):
            train_loss, train_acc= train(model, train_loader, loss, optim, args.learn, e)
        
            print("Training set: Epoch {}, Loss {}, Accuracy {}".format(e, train_loss, train_acc))
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss, test_acc, cm_valid = test(model, test_loader, loss)
            test_acc = float(test_acc)

            print("Test set: Epoch {}, Loss {}, Accuracy {}, Best Accuracy {}".format(e, valid_loss , valid_acc, acc_best))
            test_loss_list.append(valid_loss)
            test_acc_list.append(valid_acc)
            if args.dataset=="UCI":
                model.save(model, "./model_save/UCI/model" + str(valid_acc) + "_" + str(e) + ".h5")
            elif args.dataset=="wisdm":
                model.save(model, "./model_save/wisdm/model" + str(valid_acc) + "_" + str(e) + ".h5")
            elif args.dataset=="oppo":
                model.save(model, "./model_save/oppo/model" + str(valid_acc) + "_" + str(e) + ".h5")
            elif args.dataset=="Unimib":
                model.save(model, "./model_save/Unimib/model" + str(valid_acc) + "_" + str(e) + ".h5")
            else:
                model.save(model, "./model_save/pamap2/model" + str(valid_acc) + "_" + str(e) + ".h5")
    else:
        
        test_x_list = "./dataset/" + str(args.dataset) + "/x_test.npy"
        test_y_list = "./dataset/" + str(args.dataset) + "/y_test.npy"
     

        labels_name = dataset_label(args.dataset)
        print("Label names", labels_name)

        data_test = load_data(test_x_list, test_y_list)
        har_test_tensor = data_test.HAR_data()
        model = keras.models.load_model('./model_save/wisdm/model0.8505079825834543_199.h5')
        for i in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]:
            model.set_width(i)
            print("The width has been set to "+str(i))
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
            for e in range(1):
                test_loss, test_acc, conf_matrix = test(model, test_loader, loss_func)
                test_acc = float(test_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                print("Test set: Epoch {}, Loss {}, Accuracy {}".format(e, test_loss , test_acc))
                print("Test set: Epoch {}, Loss {}, Accuracy {}".format(e, test_loss, round(test_acc*100, 2)))
            plot_confusion_matrix(conf_matrix, labels_name, args.dataset) 
            gui()
        
    
   
