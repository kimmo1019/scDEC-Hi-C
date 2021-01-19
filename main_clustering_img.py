from __future__ import division
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import random
import copy
import math
import util
import metric
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import pandas as pd 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
'''
using wgan-gp
Instructions: Roundtrip model for clustering
    x,y - data drawn from base density and observation data (target density)
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping x space to y space
    H(.)  - generator network for mapping y space to x space
    Dx(.) - discriminator network in x space (latent space)
    Dy(.) - discriminator network in y space (observation space)
'''
class RoundtripModel(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, q_net, x_sampler, y_sampler, nb_classes, data, pool, batch_size, alpha, beta, is_train):
        self.data = data
        self.g_net = g_net
        self.h_net = h_net
        self.dx_net = dx_net
        self.dy_net = dy_net
        self.q_net = q_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        tf.reset_default_graph()


        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='x_onehot')
        self.x_combine = tf.concat([self.x,self.x_onehot],axis=1,name='x_combine')

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.y_ = self.g_net(self.x_combine,reuse=False)

        self.x_, self.x_onehot_, self.x_logits_ = self.h_net(self.y,reuse=False)#continuous + softmax + before_softmax
        
        self.x__, self.x_onehot__, self.x_logits__ = self.h_net(self.y_)

        self.x_combine_ = tf.concat([self.x_, self.x_onehot_],axis=1)
        self.y__ = self.g_net(self.x_combine_)

        self.dy_ = self.dy_net(self.y_, reuse=False)
        self.dx_ = self.dx_net(self.x_, reuse=False)

        self.l2_loss_x = tf.reduce_mean(tf.square(self.x - self.x__))
        self.l2_loss_y = tf.reduce_mean(tf.square(self.y - self.y__))

        #mutual information, self.x_onehot, self.prob
        self.prob = self.q_net(self.dy_,reuse=False)
        self.loss_mutual = -tf.reduce_mean(tf.reduce_sum(tf.log(self.prob + 1e-8) * self.x_onehot,axis=1))

        #self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_onehot, logits=self.x_logits__))
        self.CE_loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.x_logits__,labels=self.x_onehot))
        

        #standard gan
        #self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy_, labels=tf.ones_like(self.dy_)))
        #self.h_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx_, labels=tf.ones_like(self.dx_)))
        #wgan
        self.g_loss_adv = tf.reduce_mean(self.dy_)
        self.h_loss_adv = tf.reduce_mean(self.dx_)
        

        self.g_loss = self.g_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.h_loss = self.h_loss_adv + self.alpha*self.l2_loss_x + self.beta*self.l2_loss_y
        self.g_h_loss = self.g_loss_adv + self.h_loss_adv + self.alpha*(self.l2_loss_x + self.l2_loss_y) + self.beta*self.CE_loss_x


        # self.fake_x = tf.placeholder(tf.float32, [None, self.x_dim], name='fake_x')
        # self.fake_x_onehot = tf.placeholder(tf.float32, [None, self.nb_classes], name='fake_x_onehot')
        # self.fake_x_combine = tf.concat([self.fake_x,self.fake_x_onehot],axis=1,name='fake_x_combine')

        # self.fake_y = tf.placeholder(tf.float32, [None, self.y_dim], name='fake_y')
        
        self.dx = self.dx_net(self.x)
        self.dy = self.dy_net(self.y)

        # self.d_fake_x = self.dx_net(self.fake_x)
        # self.d_fake_y = self.dy_net(self.fake_y)

        #stardard gan
        # self.dx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dx, labels=tf.ones_like(self.dx))) \
        #     +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_x, labels=tf.zeros_like(self.d_fake_x)))
        # self.dy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dy, labels=tf.ones_like(self.dy))) \
        #     +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_y, labels=tf.zeros_like(self.d_fake_y)))
        
        #wgan
        self.dx_loss = tf.reduce_mean(self.dx) - tf.reduce_mean(self.dx_)
        self.dy_loss = tf.reduce_mean(self.dy) - tf.reduce_mean(self.dy_)

        #gradient penalty for x
        epsilon_x = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon_x * self.x + (1 - epsilon_x) * self.x_
        dx_hat = self.dx_net(x_hat)
        grad_x = tf.gradients(dx_hat, x_hat)[0] #(bs,x_dim)
        grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,)
        self.gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0)*10)

        #gradient penalty for y
        epsilon_y = tf.random_uniform([], 0.0, 1.0)
        y_hat = epsilon_y * self.y + (1 - epsilon_y) * self.y_
        dy_hat = self.dy_net(y_hat)
        grad_y = tf.gradients(dy_hat, y_hat)[0] #(bs,x_dim)
        grad_norm_y = tf.sqrt(tf.reduce_sum(tf.square(grad_y), axis=1))#(bs,)
        self.gpy_loss = tf.reduce_mean(tf.square(grad_norm_y - 1.0)*10)

        self.d_loss = self.dx_loss + self.dy_loss + self.gpy_loss + self.gpx_loss

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_h_loss, var_list=self.g_net.vars+self.h_net.vars)
        #self.d_optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr) \
        #        .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.dx_net.vars+self.dy_net.vars)
        self.l2_loss_x_optim =tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.l2_loss_x, var_list=self.h_net.vars+self.g_net.vars)
        self.l2_loss_y_optim =tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.l2_loss_y, var_list=self.h_net.vars+self.g_net.vars)

        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss_adv+self.alpha*(self.l2_loss_x)+self.beta*self.CE_loss_x, var_list=self.h_net.vars+self.g_net.vars)        
        self.h_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.h_loss_adv, var_list=self.h_net.vars+self.g_net.vars)        
        self.dx_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dx_loss+self.gpx_loss, var_list=self.dx_net.vars)
        self.dy_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(self.dy_loss+self.gpy_loss, var_list=self.dy_net.vars)

        self.mutual_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9) \
                .minimize(0.01*self.loss_mutual, var_list=self.g_net.vars+self.dy_net.vars)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.g_loss_adv_summary = tf.summary.scalar('g_loss_adv',self.g_loss_adv)
        self.h_loss_adv_summary = tf.summary.scalar('h_loss_adv',self.h_loss_adv)
        self.l2_loss_x_summary = tf.summary.scalar('l2_loss_x',self.l2_loss_x)
        self.l2_loss_y_summary = tf.summary.scalar('l2_loss_y',self.l2_loss_y)
        self.dx_loss_summary = tf.summary.scalar('dx_loss',self.dx_loss)
        self.dy_loss_summary = tf.summary.scalar('dy_loss',self.dy_loss)
        self.gpx_loss_summary = tf.summary.scalar('gpx_loss',self.gpx_loss)
        self.gpy_loss_summary = tf.summary.scalar('gpy_loss',self.gpy_loss)
        self.g_merged_summary = tf.summary.merge([self.g_loss_adv_summary, self.h_loss_adv_summary,\
            self.l2_loss_x_summary,self.l2_loss_y_summary,self.gpx_loss_summary,self.gpy_loss_summary])
        self.d_merged_summary = tf.summary.merge([self.dx_loss_summary,self.dy_loss_summary])

        #graph path for tensorboard visualization
        self.graph_dir = 'graph/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.graph_dir) and is_train:
            os.makedirs(self.graph_dir)
        
        #save path for saving predicted data
        self.save_dir = 'data/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver(max_to_keep=500)

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)


    def train(self, nb_batches, patience):
        data_y_train = copy.copy(self.y_sampler.load_all()[0])
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer=tf.summary.FileWriter(self.graph_dir,graph=tf.get_default_graph())
        start_time = time.time()
        counter = 1
        weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        for batch_idx in range(nb_batches):
        #for epoch in range(epochs):
            lr = 1e-4 #if batch_idx<10000 else 1e-4
            #update D
            for _ in range(5):
                bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
                by = self.y_sampler.train(self.batch_size)
                d_summary,_ = self.sess.run([self.d_merged_summary, self.d_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
            self.summary_writer.add_summary(d_summary,batch_idx)

            bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
            by = self.y_sampler.train(self.batch_size)
            #update G
            g_summary, _ = self.sess.run([self.g_merged_summary ,self.g_h_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
            self.summary_writer.add_summary(g_summary,batch_idx)
################################### iteratively updating ############################
            #update Dy
            # for _ in range(5):
            #     bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
            #     by = self.y_sampler.train(self.batch_size)
            #     d_summary,_ = self.sess.run([self.d_merged_summary, self.dy_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
            # self.summary_writer.add_summary(d_summary,batch_idx)

            # bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
            # by = self.y_sampler.train(self.batch_size)
            # #update G 
            # g_summary, _ = self.sess.run([self.g_merged_summary ,self.g_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
            # self.summary_writer.add_summary(g_summary,batch_idx)

            #update Dy
            # for _ in range(5):
            #     bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
            #     by = self.y_sampler.train(self.batch_size)
            #     d_summary,_ = self.sess.run([self.d_merged_summary, self.dx_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})

            # bx, bx_onehot = self.x_sampler.train(self.batch_size,weights)
            # by = self.y_sampler.train(self.batch_size)
            # #update H
            # g_summary, _ = self.sess.run([self.g_merged_summary ,self.h_optim], feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by, self.lr:lr})
################################### iteratively updating ############################
            #quick test on a random batch data
            if batch_idx % 100 == 0:
                g_loss_adv, h_loss_adv, CE_loss, l2_loss_x, l2_loss_y, g_loss, \
                    h_loss, g_h_loss, gpx_loss, gpy_loss = self.sess.run(
                    [self.g_loss_adv, self.h_loss_adv, self.CE_loss_x, self.l2_loss_x, self.l2_loss_y, \
                    self.g_loss, self.h_loss, self.g_h_loss, self.gpx_loss, self.gpy_loss],
                    feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by}
                )
                dx_loss, dy_loss, d_loss = self.sess.run([self.dx_loss, self.dy_loss, self.d_loss], \
                    feed_dict={self.x: bx, self.x_onehot: bx_onehot, self.y: by})

                print('batch_idx [%d] Time [%.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss [%.4f] gpx_loss [%.4f] gpy_loss [%.4f] \
                    l2_loss_x [%.4f] l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] dy_loss [%.4f] d_loss [%.4f]' %
                    (batch_idx, time.time() - start_time, g_loss_adv, h_loss_adv, CE_loss, gpx_loss, gpy_loss, l2_loss_x, l2_loss_y, \
                    g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
            counter += 1
            if (batch_idx+1)%1000 ==0 and (batch_idx+1)>100000:
                if batch_idx+1 == nb_batches:
                    self.evaluate(timestamp,counter,True)
                    self.save(batch_idx)
                else:
                    self.evaluate(timestamp,counter)
                    self.save(batch_idx)

                # ratio = 0.7
                # weights = ratio*weights + (1-ratio)*self.estimate_weights(use_kmeans=False)
                # weights = weights/np.sum(weights)
                # print weights


    def estimate_weights(self,use_kmeans=False):
        data_y, label_y = self.y_sampler.load_all()
        data_x_, data_x_onehot_ = self.predict_x(data_y)
        if use_kmeans:
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(np.concatenate([data_x_,data_x_onehot_],axis=1))
            label_infer = km.labels_
        else:
            label_infer = np.argmax(data_x_onehot_, axis=1)
        weights = np.empty(self.nb_classes, dtype=np.float32)
        for i in range(self.nb_classes):
            weights[i] = list(label_infer).count(i)  
        return weights/float(np.sum(weights)) 

    def evaluate(self,timestamp,batch_idx,run_kmeans=False):
        data_y, label_y, _ = self.y_sampler.load_all()
        #data_y, label_y = self.y_sampler.tst_data, self.y_sampler.tst_label
        N = data_y.shape[0]
        data_x_, data_x_onehot_ = self.predict_x(data_y)
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_x_,data_x_onehot_,label_y)
        label_infer = np.argmax(data_x_onehot_, axis=1)
        purity = metric.compute_purity(label_infer, label_y)
        nmi = normalized_mutual_info_score(label_y, label_infer)
        ari = adjusted_rand_score(label_y, label_infer)
        #self.cluster_heatmap(batch_idx, label_infer, label_y)
        print('RTM: Purity = {}, NMI = {}, ARI = {}'.format(purity,nmi,ari))
        f = open('%s/log.txt'%self.save_dir,'a+')
        f.write('%.4f\t%.4f\t%.4f\t%d\n'%(purity,nmi,ari,batch_idx))
        f.close()
        #k-means
        if run_kmeans:
            km = KMeans(n_clusters=nb_classes, random_state=0).fit(data_y)
            label_kmeans = km.labels_
            purity = metric.compute_purity(label_kmeans, label_y)
            nmi = normalized_mutual_info_score(label_y, label_kmeans)
            ari = adjusted_rand_score(label_y, label_kmeans)
            print('K-means: Purity = {}, NMI = {}, ARI = {}'.format(purity,nmi,ari))
            f = open('%s/log.txt'%self.save_dir,'a+')
            f.write('%.4f\t%.4f\t%.4f\n'%(purity,nmi,ari))
            f.close() 
    
    def cluster_heatmap(self,batch_idx,label_pre,label_true):
        assert len(label_pre)==len(label_true)
        confusion_mat = np.zeros((self.nb_classes,self.nb_classes))
        for i in range(len(label_true)):
            confusion_mat[label_pre[i]][label_true[i]] += 1
        #columns=[item for item in range(1,11)]
        #index=[item for item in range(1,11)]
        #df = pd.DataFrame(confusion_mat,columns=columns,index=index)
        plt.figure()
        df = pd.DataFrame(confusion_mat)
        sns.heatmap(df,annot=True, cmap="Blues")
        plt.savefig('%s/heatmap_%d.png'%(self.save_dir,batch_idx))
        plt.close()


    #predict with y_=G(x)
    def predict_y(self, x, x_onehot, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x_onehot = x_onehot[ind, :]
            batch_y_ = self.sess.run(self.y_, feed_dict={self.x:batch_x, self.x_onehot:batch_x_onehot})
            y_pred[ind, :] = batch_y_
        return y_pred
    
    #predict with x_=H(y)
    def predict_x(self,y,bs=256):
        assert y.shape[-1] == self.y_dim
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        x_onehot = np.zeros(shape=(N, self.nb_classes)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_x_,batch_x_onehot_ = self.sess.run([self.x_, self.x_onehot_], feed_dict={self.y:batch_y})
            x_pred[ind, :] = batch_x_
            x_onehot[ind, :] = batch_x_onehot_
        return x_pred, x_onehot


    def save(self,batch_idx):

        checkpoint_dir = 'checkpoint/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'),global_step=batch_idx)

    def load(self, pre_trained = False, timestamp='',batch_idx=999):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_{}_{}}'.format(self.data, self.x_dim,self.y_dim, self.alpha, self.beta)
        else:
            if timestamp == '':
                print('Best Timestamp not provided.')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint/cluster_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt-%d'%batch_idx))
                print('Restored model weights.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--K', type=int, default=11)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--nb_batches', type=int, default=100000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()
    data = args.data
    model = importlib.import_module(args.model)
    nb_classes = args.K
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    epochs = args.epochs
    nb_batches = args.nb_batches
    patience = args.patience
    alpha = args.alpha
    beta = args.beta
    timestamp = args.timestamp
    is_train = args.train
    g_net = model.Generator_img(nb_classes=nb_classes,output_dim = y_dim,name='g_net',nb_layers=2,nb_units=64,dataset=data,is_training=True)
    h_net = model.Encoder_img(nb_classes=nb_classes,output_dim = x_dim+nb_classes,name='h_net',nb_layers=2,nb_units=64,dataset=data,cond=True)
    dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=64)
    dy_net = model.Discriminator_img(input_dim=y_dim,name='dy_net',nb_layers=2,nb_units=128,dataset=data)
    q_net = model.MutualNet(output_dim=nb_classes, name='mutual_net',nb_units=128)
    
    #model from ClusterGAN
    # g_net = model.Generator_img(x_dim = y_dim)
    # h_net = model.Encoder_img(z_dim=x_dim+nb_classes, dim_gen = x_dim)
    # dx_net = model.Discriminator(input_dim=x_dim,name='dx_net',nb_layers=2,nb_units=64)
    # dy_net = model.Discriminator_img()
   
    pool = util.DataPool(10)

    xs = util.Mixture_sampler(nb_classes=nb_classes,N=10000,dim=x_dim,sd=1,scale=0.1)
    ys = util.scHiC_sampler()

    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, q_net, xs, ys, nb_classes, data, pool, batch_size, alpha, beta, is_train)

    if args.train:
        RTM.train(nb_batches=nb_batches, patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, batch_idx = nb_batches-1)
            
