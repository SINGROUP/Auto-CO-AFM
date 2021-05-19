
# Several service functions

import os
import sys
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nimg

from PIL import Image
from scipy import misc

from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapz
from scipy.spatial.distance import cdist, pdist
from itertools import product

from keras import backend as K
import pickle

def _calc_plot_dim(n, f=0.3):
    rows = max(int(np.sqrt(n) - f), 1)
    cols = 1
    while rows*cols < n:
        cols += 1
    return rows, cols




class LossLog:

    def __init__(self, log_dir='./', log_path=None, descriptors=None):

        self.log_dir = log_dir
        if log_path is None:
            self.log_path = log_dir+'loss_log.csv'
        else:
            self.log_path = log_path
        self.descriptors = descriptors
        self.train_loss = None
        self.val_loss = None
        self.epoch = 0

        self._init_log()

    def _init_log(self):
        if not(os.path.isfile(self.log_path)):
            if self.descriptors is None:
                raise ValueError('No loss names given and no pre-existing log found')
            self.descriptors = [l.replace(' ','_') for l in self.descriptors]
            with open(self.log_path, 'w') as f:
                f.write('epoch')
                for descriptor in self.descriptors:
                    f.write(';'+descriptor)
                for descriptor in self.descriptors:
                    f.write(';val_'+descriptor)
                f.write('\n')
            print('Created log at '+str(self.log_path))
        else:
            with open(self.log_path, 'r') as f:
                header = f.readline().rstrip('\r\n').split(';') 
                hl = len(header)-1
                if self.descriptors is None:
                    self.descriptors = [l.replace(' ','_') for l in header[1:hl/2+1]]
                elif len(self.descriptors) != hl/2:
                    raise ValueError('The length of the given list of loss names and the length of the header of the existing log at '+self.log_path+' do not match.')
                for line in f:
                    line = line.rstrip('\n').split(';')
                    loss = [float(s) for s in line[1:]]
                    self._add_loss_to_array(loss)
            print('Using existing log at '+str(self.log_path))
                
    def _add_loss_to_array(self, added_loss):
        ll = len(self.descriptors)
        added_train_loss =  np.expand_dims(added_loss[:ll], axis=0)
        added_val_loss = np.expand_dims(added_loss[ll:], axis=0)
        if self.train_loss is None:
            self.train_loss = added_train_loss
            self.val_loss = added_val_loss
        else:
            self.train_loss = np.append(self.train_loss, added_train_loss, axis=0)
            self.val_loss = np.append(self.val_loss, added_val_loss, axis=0)
        self.epoch += 1

    def _write_loss_to_log(self, added_loss):
        ll = len(self.descriptors)
        added_train_loss = added_loss[:ll]
        added_val_loss = added_loss[ll:]
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch))
            for i in range(len(self.descriptors)):
                f.write(';'+str(added_train_loss[i]))
            for i in range(len(self.descriptors)):
                f.write(';'+str(added_val_loss[i]))
            f.write('\n')

    def add_loss(self, loss):
        self._add_loss_to_array(loss)
        self._write_loss_to_log(loss)

    def plot_history(self, out_files=None, save=True, show=False):

        if out_files is None:
            out_files = []
            for descriptor in self.descriptors:
                out_files.append(self.log_dir+'loss_history_'+descriptor+'.png')
        else:
            if len(out_files) != len(self.descriptors):
                raise ValueError('The length of the output file list does not match the number of losses to plot')

        x = range(1,self.epoch+1)

        for i in range(len(self.descriptors)):

            plt.figure
            plt.semilogy(x,self.train_loss[:,i])
            plt.semilogy(x,self.val_loss[:,i])
            plt.legend(['Training', 'Validation'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            if save:
                plt.savefig(out_files[i])
                print('Loss history plot saved to '+out_files[i])
            
            if show:
                plt.show()
            else:
                plt.close()

class LossAccuLog:

    def __init__(self, log_dir='./', log_path=None, accu_path=None,descriptors=None):

        self.log_dir = log_dir
        if log_path is None:
            self.log_path = log_dir+'loss_log.csv'
        else:
            self.log_path = log_path
        if accu_path is None:
            self.accu_path = log_dir+'accu_log.csv'

        else:
            self.accu_path = accu_path
        self.descriptors = descriptors
        self.train_loss = None
        self.val_loss = None
        self.train_accu = None
        self.val_accu = None
        self.epoch = 0

        self._init_loss()
        self._init_accu()
    def _init_loss(self):
        if not(os.path.isfile(self.log_path)):
            if self.descriptors is None:
                raise ValueError('No loss names given and no pre-existing log found')
            self.descriptors = [l.replace(' ','_') for l in self.descriptors]
            with open(self.log_path, 'w') as f1:
                f1.write('epoch')
                for descriptor in self.descriptors:
                    f1.write(';'+descriptor)
                for descriptor in self.descriptors:
                    f1.write(';val_'+descriptor)
                f1.write('\n')
            print('Created log loss at '+str(self.log_path))

            

        else:
            with open(self.log_path, 'r') as f:
                header = f.readline().rstrip('\r\n').split(';') 
                hl = len(header)-1
                if self.descriptors is None:
                    self.descriptors = [l.replace(' ','_') for l in header[1:hl/2+1]]
                elif len(self.descriptors) != hl/2:
                    raise ValueError('The length of the given list of loss names and the length of the header of the existing log at '+self.log_path+' do not match.')
                for line in f:
                    line = line.rstrip('\n').split(';')
                    loss = [float(s) for s in line[1:]]
                    self._add_loss_to_array(loss)
            print('Using existing log loss at '+str(self.log_path))
            
    def _init_accu(self):         
        if not(os.path.isfile(self.accu_path)):
            with open(self.accu_path, 'w') as f2:
                f2.write('epoch')
                for descriptor in self.descriptors:
                    f2.write(';'+descriptor)
                for descriptor in self.descriptors:
                    f2.write(';val_'+descriptor)
                f2.write('\n')
            print('Created log accu at '+str(self.accu_path))
        else:
            with open(self.accu_path, 'r') as f:
                header = f.readline().rstrip('\r\n').split(';') 
                hl = len(header)-1
                for line in f:
                    line = line.rstrip('\n').split(';')
                    accu = [float(s) for s in line[1:]]
                    self._add_accu_to_array(accu)
            print('Using existing log accu at '+str(self.accu_path))

    def _add_loss_to_array(self, added_loss):
        ll = len(self.descriptors)
        added_train_loss =  np.expand_dims(added_loss[:ll], axis=0)
        added_val_loss = np.expand_dims(added_loss[ll:], axis=0)
        if self.train_loss is None:
            self.train_loss = added_train_loss
            self.val_loss = added_val_loss
        else:
            self.train_loss = np.append(self.train_loss, added_train_loss, axis=0)
            self.val_loss = np.append(self.val_loss, added_val_loss, axis=0)
        self.epoch += 1
    
    def _add_accu_to_array(self, added_accu):
        ll = len(self.descriptors)
        added_train_accu =  np.expand_dims(added_accu[:ll], axis=0)
        added_val_accu = np.expand_dims(added_accu[ll:], axis=0)
        if self.train_accu is None:
            self.train_accu = added_train_accu
            self.val_accu = added_val_accu
        else:
            self.train_accu = np.append(self.train_accu, added_train_accu, axis=0)
            self.val_accu = np.append(self.val_accu, added_val_accu, axis=0)
 
    def _write_loss_to_log(self, added_loss):
        ll = len(self.descriptors)
        added_train_loss = added_loss[:ll]
        added_val_loss = added_loss[ll:]
        with open(self.log_path, 'a') as f1:
            f1.write(str(self.epoch))
            for i in range(len(self.descriptors)):
                f1.write(';'+str(added_train_loss[i]))
            for i in range(len(self.descriptors)):
                f1.write(';'+str(added_val_loss[i]))
            f1.write('\n')
    def _write_accu_to_log(self, added_accu):
        ll = len(self.descriptors)
        added_train_accu = added_accu[:ll]
        added_val_accu = added_accu[ll:]
        with open(self.accu_path, 'a') as f2:
            f2.write(str(self.epoch))
            for i in range(len(self.descriptors)):
                f2.write(';'+str(added_train_accu[i]))
            for i in range(len(self.descriptors)):
                f2.write(';'+str(added_val_accu[i]))
            f2.write('\n')

    def add_loss(self, loss):
        self._add_loss_to_array(loss)
        self._write_loss_to_log(loss)

    def add_accu(self, accu):
        self._add_accu_to_array(accu)
        self._write_accu_to_log(accu)

    def plot_history_loss(self, out_files_loss=None, save=True, show=False):

        if out_files_loss is None:
            out_files_loss = []
            for descriptor in self.descriptors:
                out_files_loss.append(self.log_dir+'loss_history_'+descriptor+'.png')
        else:
            if len(out_files_loss) != len(self.descriptors):
                raise ValueError('The length of the output file list does not match the number of losses to plot')

        x = range(1,self.epoch+1)
             
        for i in range(len(self.descriptors)):
 
            plt.figure
            plt.semilogy(x,self.train_loss[:,i])
            plt.semilogy(x,self.val_loss[:,i])
            plt.legend(['Training', 'Validation'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            if save:
                plt.savefig(out_files_loss[i])
                print('Loss history plot saved to '+out_files_loss[i])
            
            if show:
                plt.show()
            else:
                plt.close()

    def plot_history_accu(self, out_files_accu=None, save=True, show=False):

        if out_files_accu is None:
            out_files_accu = []
            for descriptor in self.descriptors:
                out_files_accu.append(self.log_dir+'accu_history_'+descriptor+'.png')
        else:
            if len(out_files_accu) != len(self.descriptors):
                raise ValueError('The length of the output file list does not match the number of accus to plot')

        x = range(1,self.epoch+1)

        for i in range(len(self.descriptors)):
 
            plt.figure
            plt.plot(x,self.train_accu[:,i])
            plt.plot(x,self.val_accu[:,i])
            plt.legend(['Training', 'Validation'])
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            if save:
                plt.savefig(out_files_accu[i])
                print('Accuracy history plot saved to '+out_files_accu[i])
            
            if show:
                plt.show()
            else:
                plt.close()



def calculate_losses(model, true, preds=None, X=None):

    import keras.backend as K

    if preds is None and X is None:
        raise ValueError('preds and X cannot both be None')
    
    if preds is None:
        preds = model.predict_on_batch(X)

    if not isinstance(true, list):
        true = [true]
    if not isinstance(preds, list):
        preds = [preds]
    
    losses = np.zeros((true[0].shape[0], len(true)))
    for i, (t, p) in enumerate(zip(true, preds)):
        t = K.variable(t)
        p = K.variable(p) 
        loss = model.loss_functions[i](t, p)
        sh = loss.shape.as_list()
        if len(sh) > 1:
            loss = K.mean(K.reshape(loss, (sh[0],-1)), axis=1)
        losses[:,i] = K.eval(loss)

    if losses.shape[1] == 1:
        losses = losses[:,0]
    if losses.shape[0] == 1 and losses.ndim == 1:
        losses = losses[0]
    
    return losses

class LossCalculator:
    '''
    Calculates the losses for each prediction of a model
    '''

    def __init__(self, model):
        self.model = model
        self._make_loss_functions()

    def _make_loss_functions(self):
        import keras.backend as K
        self.loss_funs = []
        for i, lf in enumerate(self.model.loss_functions):
            t = K.placeholder(shape=self.model.outputs[i].shape)
            p = K.placeholder(shape=self.model.outputs[i].shape)
            self.loss_funs.append(K.function([t,p], [lf(t, p)]))

    def __call__(self, true, preds=None, X=None):
        '''
        Arguments: 
            true: Reference outputs
            preds: (optional) Predicted outputs
            X: (optional) Inputs that will be used for making predictions if preds == None
        Note: At least one of preds or X has to be provided
        '''

        if preds is None:
            if X is None:
                raise ValueError('preds and X cannot both be None')
            else:
                preds = self.model.predict_on_batch(X)

        if not isinstance(true, list):
            true = [true]
        if not isinstance(preds, list):
            preds = [preds]
        
        losses = np.zeros((true[0].shape[0], len(true)))
        for i, (t, p) in enumerate(zip(true, preds)):
            loss = self.loss_funs[i]([t,p])[0]
            sh = loss.shape
            if len(sh) > 1:
                loss = np.mean(loss.reshape((sh[0],-1)), axis=1)
            losses[:,i] = loss
        
        if losses.shape[1] == 1:
            losses = losses[:,0]
        if losses.shape[0] == 1 and losses.ndim == 1:
            losses = losses[0]

        return losses


def plot_model_layers(model, X, outdir='./layer_preds/', verbose=1):

    from keras.models import Model
    from keras.utils import plot_model

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not isinstance(X, list):
        X = [X]

    savefile = outdir+'0_model.png'
    plot_model(model, to_file=savefile)     # <- This requires pydot
    print('Model plot saved to '+savefile)

    inp = model.inputs
    if X[0].shape == inp[0].shape[1:]:
        X = [np.expand_dims(x, axis=0) for x in X]
    elif X[0].shape[0] > 1:
        X = [np.expand_dims(x[0], axis=0) for x in X]

    outputs = []
    layers = []
    for layer in model.layers[len(inp):]:
        layers.append(layer.name)
        if len(layer._inbound_nodes) == 1:
            outputs.append(layer.output)
        else:
            outputs.append(layer.get_output_at(1))
        
    dummy_model = Model(inputs=inp, outputs=outputs)
    preds = dummy_model.predict(X)
    
    for num, pred in enumerate(preds):
        
        pred = pred[0]
        if pred.ndim == 1:
            x,y = _calc_plot_dim(len(pred))
            pred = np.pad(pred, (0, x*y - len(pred)), 'constant', constant_values=np.nan)
            pred = pred.reshape((x,y))
        if pred.ndim == 2:
            pred = np.expand_dims(pred, axis=-1)
        if pred.ndim == 3:
            pred = np.expand_dims(pred, axis=-1)
        
        for i in range(pred.shape[3]):

            rows, cols = _calc_plot_dim(pred.shape[2])
            if rows == cols == 1:
                figsize=(5.0, 5.0)
            else:
                figsize=(2.5*cols,2.5*rows)
            fig = plt.figure(figsize=figsize)
            current_cmap = cm.get_cmap()
            current_cmap.set_bad('white',1.0)
            
            pi = pred[:,:,:,i]
            masked_array = np.ma.array(pi, mask=np.isnan(pi))
            vmax = masked_array.flatten().max()
            vmin = masked_array.flatten().min()
            for j in range(pred.shape[2]):
                ax = fig.add_subplot(rows,cols,j+1)
                im1 = ax.imshow(pi[:,:,j], vmax=vmax, vmin=vmin, cmap=current_cmap)
            
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            fig.colorbar(im1, cax=cbar_ax)

            save_name = outdir+'layer'+str(num)+'_'+layers[num]
            if pred.shape[-1] > 1:
                save_name += '_channel'+str(i)
            save_name += '.png'
            plt.savefig(save_name, bbox_inches="tight")
            plt.close()
            if verbose > 0: print('Visualization layer saved to '+save_name)

def divide_to_classes(xyz_batch, classes):
    '''
    Divide atomic numbers to classes
    '''

    new_batch = xyz_batch.copy()
    if classes == [None]:
        new_batch[:,:,3] = 1
        return new_batch, 2
    
    n_classes = len(classes)
    for mol in new_batch:
        for atom in mol:
            if atom[3] == 0:
                continue
            class_found = False
            for i, cls in enumerate(classes):
                if atom[3] in cls:
                    class_found = True
                    atom[3] = i+1
            if not class_found:
                atom[3] = len(classes)+1
                n_classes = len(classes)+1
    return new_batch, n_classes+1


class accuMetrics:
    
    def __init__(self, n_classes,  conf_level_points=101, conf_mat_level=0.5):
        self.n_classes = n_classes
        self.conf_level_points = conf_level_points
        self.conf_mat_level = conf_mat_level
        self.conf_mat = {}
 
        for conf_level in np.linspace(0, 1, conf_level_points):
            self.conf_mat[conf_level] = np.zeros((n_classes, n_classes))
        if conf_mat_level not in self.conf_mat:
            self.conf_mat[conf_mat_level] = np.zeros((n_classes, n_classes))

            
        self.dist = []
        self.losses = []

    def add_preds(self, preds, true, losses):
        '''
        Add a batch of predictions.
        Arguments:
            preds: np.ndarray in same format as the return value of encode_xyz. Predictions.
            true: np.ndarray in same format as the return value of encode_xyz. References.
            losses: np.ndarray of shape (1, batch_size). Losses for each batch item.
        '''
         
        #preds = preds.reshape((-1, preds.shape[-1]))
        #true = true.reshape((-1, preds.shape[-1]))
    
        # Add to confusion matrix
        true_classes = true.astype(int)
        for conf_level in self.conf_mat.keys():

            pred_classes = np.ones_like(true_classes) #np.argmax(preds, axis=1)
            pred_classes[preds < conf_level] = 0

            np.add.at(self.conf_mat[conf_level], (true_classes, pred_classes), 1)

 
    def plot(self, outdir='./', verbose=1):
        
        # Statistics

        self.acc = np.sum(np.diagonal(self.conf_mat[self.conf_mat_level])) / np.sum(self.conf_mat[self.conf_mat_level])
        self.prec = np.zeros(self.n_classes)
        self.rec = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            self.prec[i] = self.conf_mat[self.conf_mat_level][i,i] / np.sum(self.conf_mat[self.conf_mat_level][:,i])
            self.rec[i] = self.conf_mat[self.conf_mat_level][i,i] / np.sum(self.conf_mat[self.conf_mat_level][i,:])

 
        outfile = outdir+'metrics.csv'
        with open(outfile, 'w') as f:
 
            f.write('Accuracy:;%f\n' % self.acc)
            f.write('Class;Precision;Recall;Confusion matrix\n')
            for i in range(self.n_classes):
                f.write('%d;%f;%f;' % (i, self.prec[i], self.rec[i]))
                for j in range(self.n_classes):
                    f.write(str(self.conf_mat[self.conf_mat_level][i,j]))
                    if j < self.n_classes - 1:
                        f.write(';')
                f.write('\n')
            f.write('%s;%f;%f\n\n' % ('Mean', np.mean(self.prec), np.mean(self.rec)))
     
                
        if verbose > 0: print('Statistical information saved to '+outfile)

        fs = 16
        
        
        # Precision-Recall curve
        
        self.prec_threshold = []
        self.rec_threshold = []
 
        self.auc = []
        for i in range(self.n_classes):
            pi = []
            ri = []
 
            conf_levels = sorted(self.conf_mat.keys())
            for j, conf_level in enumerate(conf_levels):
                if (np.sum(self.conf_mat[conf_level][:,i]) >0 ):
                    pi += [self.conf_mat[conf_level][i,i] / np.sum(self.conf_mat[conf_level][:,i])]
                else:
                    pi.append(1.0)
                if (np.sum(self.conf_mat[conf_level][i,:]) >0 ):                 
                    ri += [self.conf_mat[conf_level][i,i] / np.sum(self.conf_mat[conf_level][i,:])]
                else:
                    ri.append(1.0)
                 
                if np.isnan(pi[-1]):
                    pi[-1] = 1.0
 
                if conf_level == self.conf_mat_level:
                    conf_mat_ind = j
            
            if i == 0:
                cls = 'bads'
            else:
                cls = 'goods' 
            
            fig = plt.figure()
            fig.set_size_inches(12,6)
            ax1 = fig.add_axes([0.05, 0.1,0.4,0.8])
            ax2 = fig.add_axes([0.55, 0.1,0.4,0.8])

            ax1.plot(ri, pi)
            ax2.plot(conf_levels, pi)
            ax2.plot(conf_levels, ri)
            ax1.scatter(ri[conf_mat_ind], pi[conf_mat_ind], marker='o', color='r')

            ax1.set_title('Precision-Recall curve')
            ax1.set_xlabel('Recall', fontsize=12)
            ax1.set_ylabel('Precision', fontsize=12)
            ax2.set_title('Precision and Recall')
            ax2.set_xlabel('Confidence threshold', fontsize=12)
            ax2.legend(['Precision', 'Recall'])

            # Area under the curve
            auc = -trapz(pi, ri)
            fig.text(0.06, 0.12, 'AUC: %.4f' % auc, fontsize=fs)

            # Store values
            self.prec_threshold.append(pi)
            self.rec_threshold.append(ri)
            
            self.auc.append(auc)
            
            # Save figure
            outfile = outdir+'prec-rec_%s.png' % cls
            plt.savefig(outfile)
            if verbose > 0: print('precision-recall curve (%s) plot saved to %s' % (cls, outfile))
            plt.close()

        # Save data to text file
        with open(outdir+'prec_rec.csv', 'w') as f:
            for i in range(self.n_classes):
                f.write('Class %d\n' % i)
                for prec, rec in zip(self.prec_threshold[i], self.rec_threshold[i]):
                    f.write('%f,%f\n' % (prec, rec))
        
        



def add_norm_CO(X_):
    sh = X_.shape

    for j in range(sh[0]):
        mean=np.mean(X_[j,:,:])            
        sigma=np.std(X_[j,:,:])          
        X_[j,:,:]-= mean
        X_[j,:,:]= X_[j,:,:]/ sigma

def generate_data_from_images(path_to_data,dir,classes, crop_size, enable_rotations=True, enable_flips=True):
    #enable_rotations=True means that each image will be augumented by 3 rotations: 90, 180, 270 degrees.

    dataX = []
    dataY = []
    z = len(classes)
    if (enable_rotations):
        amount_rot = 4
    else:
        amount_rot = 1

    for class_ind, cls in enumerate(classes):

        images = glob.glob(path_to_data+'/'+dir+'/'+cls + '/' +'*.png')
        for image_path in images:
            image = np.array((Image.open(image_path).resize((crop_size,crop_size), Image.ANTIALIAS))).astype(np.float32)
            #print (image.shape) 
            for rotations in range(amount_rot): 
                    rot_image = np.rot90(image,rotations, axes=(-2,-1))             
                    dataX.append(rot_image)
                    dataY.append(class_ind)
                    if (enable_flips):
                        dataX.append(np.flipud(rot_image))
                        dataY.append(class_ind)
                        #dataX.append(np.fliplr(rot_image))
                        #dataY.append(class_ind)
            
    dataX = np.array(dataX)
    dataY = np.array( dataY)
    return dataX , dataY




def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

def plot_confusion_matrix(ax, conf_mat, tick_labels):
    conf_mat_norm = np.zeros_like(conf_mat, dtype=np.float64)
    for i, r in enumerate(conf_mat):
        conf_mat_norm[i] = r / np.sum(r)
    im = ax.imshow(conf_mat_norm, cmap=cm.Blues )
    plt.colorbar(im)
    ax.set_xticks(np.arange(conf_mat.shape[0]))
    ax.set_yticks(np.arange(conf_mat.shape[1]))
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels, rotation='vertical', va='center')
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            color = 'white' if conf_mat_norm[i,j] > 0.5 else 'black'
            label = '{:.3f}'.format(conf_mat_norm[i,j])+'\n('+'{:d}'.format(conf_mat[i,j])+')'
            ax.text(j, i, label, ha='center', va='center', color=color)
    plt.ylim([1.5, -0.5])
    plt.tight_layout()


def plot_roc_curve(ax, fpr, tpr,roc_auc):
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    plt.tight_layout()
 
def make_batch_plots(X,Y, preds, epoch,batch_ind, set_name, 
    classes          = ['bads', 'goods'], 
    cmap=cm.gray,outdir = './CNN/', verbose = 1):
    outdir = outdir+ 'predictions/'
    column_names = ['AFM Data']    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    img_ind = 0
     
    num_rows = 1
    num_cols = 8
    rows, cols = num_rows, num_cols  #X.shape[0], X.shape[1]
    permut = np.array( range(X.shape[0]) )
    np.random.shuffle( permut )
    fig = plt.figure(figsize=(4*cols,4*rows))
    for i in range(num_rows):
        for j in range(num_cols):
            fig.add_subplot(rows,cols,img_ind+1)
            plt.imshow(X[img_ind*4,0,:,:], cmap = cmap, origin="lower")
            plt.colorbar()      
            formattedList = [f'{preds[img_ind].tolist(): 0.2f}',f'{1.-preds[img_ind].tolist(): 0.2f}'] 
            plt.xlabel(f'{formattedList} True:{classes[Y[img_ind*4]]}') 
            img_ind+=1
    save_name = f'{outdir}epoch{epoch}_batch{batch_ind}_{set_name}.png'
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()
    if verbose > 0: print('Input image saved to '+save_name)


def save_optimizer_state(model, save_path):
    '''
    Save keras optimizer state.
    Arguments:
        model: tensorflow.keras.Model.
        save_path: str. Path where optimizer state file is saved to.
    '''
    #weights = model.optimizer.get_weights()
    #np.savez(save_path, weights)
    #print(f'Optimizer weights saved to {save_path}')

    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_path, 'wb') as f:
        pickle.dump(weight_values, f)
    print(f'Optimizer weights saved to {save_path}')




def load_optimizer_state(model, load_path):
    '''
    Load keras optimizer state.
    Arguments:
        model: tensorflow.keras.Model.
        save_path: str. Path where optimizer state file is loaded from.
    '''

    if not os.path.exists(load_path):
        print(f'No optimizer weights found at {load_path}')
        return 0

    model._make_train_function()
    with open(load_path, 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)
    print(f'Optimizer weights loaded from {load_path}')

    return 1
    

