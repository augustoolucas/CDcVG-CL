__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import os
import sys
import glob
import pickle
import torchvision
import time
from torchvision import transforms
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from mllogger import *
from datasets import CIFAR10Ex, CelebAEx, MNISTEx
from models5 import *
from sys_utils import *
from image_utils import *
from sampler import sample, interpolate_rnd, get_covb

# ===================================================================================
class Solver():
    def __init__(self, hps, logr):
        self.hps = hps
        self.logr = logr
        self.mse = torch.nn.MSELoss()
        self.G = None
        self.E = None
        self.current_best = 1e+30
        return

    def save_checkpoint(self, filename=None, epoch=0, iter=0, current_loss=None):

        def add_to_state(model, name):
            if self.hps.parallel:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            states.update({name:state_dict})

        if filename is None:
            filename=self.logr.model_path+'/weights-'+str(epoch)+'.cp'

        path,_ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        states = {'epoch':epoch,
                'iter': iter,
                'loss_eval': current_loss }

        if self.G is not None:
            add_to_state(self.G, 'gen')
        if self.E is not None:
            add_to_state(self.E, 'enc')

        with open(filename, mode='wb+') as f:
            torch.save(states, f)

        os.system('cp '+filename +' '+ self.logr.model_path+'/last.cp')

        # Save the best model
        if current_loss is not None and current_loss < self.current_best:
            os.system('cp '+filename +' '+ self.logr.model_path+'/best.cp')
            self.current_best  = current_loss
        
        # Purge old ones
        if self.hps.keep_last_models is not None:
            files = [f for f in os.listdir(self.logr.model_path) if 'weights-' in f]
            wfiles = []
            for f in files:
                fname,_ = os.path.splitext(f)
                epoch = fname.split('-')[-1] 
                wfiles.append([f, int(epoch)])

            wfiles.sort(key=lambda x: x[1])
            wfiles.reverse()
            to_remove = wfiles[self.hps.keep_last_models:]
            for f in to_remove:
                filename = os.path.join(self.logr.model_path, f[0])
                os.system('rm '+filename)

        return

    def load_checkpoint(self, filename=None): 
        epoch=-1
        iter = -1
        loss_eval = -1

        if filename is None:
            filename=self.logr.model_path+'/last.cp'

        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
            epoch = checkpoint['epoch']
            iter = checkpoint.get('iter', -1)
            loss_eval = checkpoint.get('loss_eval', -1)
            if 'gen' in checkpoint:
                self.G.load_state_dict(checkpoint['gen'], strict=False)
            if 'enc' in checkpoint:
                self.E.load_state_dict(checkpoint['enc'], strict=False)

            print("=> loaded checkpoint '{} (epoch {},  loss {})'".format(filename, epoch, loss_eval))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        self.hps.epoch_start = epoch
        self.hps.iter_start = iter
        self.current_best = loss_eval
        return epoch


    def load_data(self):
        dataroot = "~/projects/data/"

        if self.hps.dataset=='celeba':
            if self.hps.img_crop_size is not None:
                transform = transforms.Compose([
                    transforms.CenterCrop(self.hps.img_crop_size),
                    transforms.Resize(self.hps.img_size),
                    transforms.ToTensor(),
                    #    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.hps.img_size),
                    # smaller edge of the image will be matched to this number. 
                    # i.e, if height > width, then image will be rescaled to (size * height / width, size)
                    transforms.CenterCrop(self.hps.img_size),
                    transforms.ToTensor(),
                    #    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    ])

            self.train_dataset= CelebAEx(dataroot+"CelebA/", split='train', download=True, transform=transform, 
                    corrupt_method=self.hps.corrupt_method, corrupt_args=self.hps.corrupt_args)

            self.test_dataset = CelebAEx(dataroot+"CelebA/", split='test', download=True, transform=transform, 
                    corrupt_method=self.hps.corrupt_method, corrupt_args=self.hps.corrupt_args)


        elif self.hps.dataset == 'mnist':
            transform = transforms.Compose([
                # transforms.Resize(self.hps.img_size),
                transforms.Pad(2, fill=0),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            self.train_dataset = MNISTEx(dataroot+'MNIST', train=True, download=True, transform=transform,
                    corrupt_method=self.hps.corrupt_method, corrupt_args=self.hps.corrupt_args)
            self.test_dataset = MNISTEx(dataroot+'MNIST', train=False, download=True, transform=transform,
                    corrupt_method=self.hps.corrupt_method, corrupt_args=self.hps.corrupt_args)
            indexes = self.train_dataset.targets < 2
            self.train_dataset.data = self.train_dataset.data[indexes]
            self.train_dataset.targets = self.train_dataset.targets[indexes]
            indexes = self.test_dataset.targets < 2
            self.test_dataset.data = self.test_dataset.data[indexes]
            self.test_dataset.targets = self.test_dataset.targets[indexes]


        elif self.hps.dataset == 'cifar10':
            transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(self.hps.img_size),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])
            self.train_dataset = CIFAR10Ex(dataroot+'cifar10', train=True, transform=transform, download=True,
                    corrupt_method=self.hps.corrupt_method, corrupt_args=self.hps.corrupt_args)
            self.test_dataset = CIFAR10Ex(dataroot+'cifar10', train=False, transform=transform, download=True,
                    corrupt_method=self.hps.corrupt_method_test, corrupt_args=self.hps.corrupt_args_test)

        else:
            print("Wrong dataset name:", self.hps.dataset)
            sys.exit(0)
        

        print('Training size:', len(self.train_dataset)) 
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hps.batch_size,
                shuffle=True, num_workers=self.hps.workers, drop_last=True, pin_memory=True)

        self.test_dataloader = None
        if self.test_dataset is not None:
            print('Test size:', len(self.test_dataset)) 
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, 
                    batch_size=int(self.hps.batch_size_test),
                    shuffle=False, num_workers=8, drop_last=False, pin_memory=True)


        self.logr.set_samples_num('Train', len(self.train_dataloader.dataset))
        if self.test_dataloader is not None:
            self.logr.set_samples_num('Eval', len(self.test_dataloader.dataset))
        return


    def net_init(self):
        if self.hps.use_cuda:
            if not self.hps.parallel and self.hps.cuda_device > -1:
                print("Setting CUDA device: ", self.hps.cuda_device)
                torch.cuda.set_device(int(self.hps.cuda_device))

        self.E = None
        self.G = None
        #self.Classifier = Classifier(self.hps).to('cuda')
        self.Discriminator = Discriminator(self.hps).to('cuda')

        # Select model by model name
        if self.hps.vae_model is not None:
            gen_net_name = 'Gen'+self.hps.vae_model
            enc_net_name = 'Enc'+self.hps.vae_model

            net_class = globals()[gen_net_name]
            self.G = net_class(self.hps)
            weight_init(self.G)

            net_class = globals()[enc_net_name]
            self.E = net_class(self.hps)
            weight_init(self.E)
        
        else:
            print("No VAE model specified! Running wihout VAE", self.hps.vae_model)
            # sys.exit(0)

        if self.G is not None and self.E is not None:
            print("Encoder:")
            net_info(self.E)
            print("Generator:")
            net_info(self.G)

            summary(self.E.cuda(), (self.hps.channels, self.hps.img_size, self.hps.img_size))
            summary(self.G.cuda(), (1,self.hps.zsize + 10))
        return

    def reparam_log(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).cuda()
        z = mu + eps*std
        return z

    def train(self):
        if self.hps.parallel:
            self.E = nn.DataParallel(self.E)
            self.G = nn.DataParallel(self.G)

        if self.G is not None and self.E is not None:
            params = list(self.E.parameters()) + list(self.G.parameters())
            self.optim = torch.optim.Adam(params=params , lr=self.hps.lr[0], weight_decay=self.hps.l2 )
            #self.optim_C = torch.optim.Adam(params=self.Classifier.parameters(), lr=self.hps.lr[0])
            self.optim_D = torch.optim.Adam(params=self.Discriminator.parameters(), lr=self.hps.lr[0] * .1, weight_decay=0.001)

        z_static = torch.randn(self.hps.batch_size, self.hps.zsize)
        if self.hps.use_cuda:
            if self.G is not None and self.E is not None:
                self.G.cuda()
                self.E.cuda()
            z_static =z_static.cuda()

        iter = self.hps.iter_start+1
        self.logr.iter_global = iter
        self.ws = None
        if self.hps.epoch_start is not None:
            # print out config diff between the epoch we are resuming and the current config
            past_cfg=self.logr.load_config()
            if past_cfg is not None:
                cfg_diff = self.hps.diff_str(past_cfg)
                print("\n** Configuration DIFF:")
                print(cfg_diff)

            print("Evaluating at epoch: ", self.hps.epoch_start, flush=True)
            self.eval_reconstruct(dataset=self.test_dataloader, at_epoch=self.hps.epoch_start, iter=iter)
            pass
        else:
            self.hps.epoch_start= -1

        start_from_epoch = self.hps.epoch_start+1
        self.logr.save_config(epoch=start_from_epoch)
        print("Starting training from epoch=",start_from_epoch,"  iter=",iter)

        mse = nn.MSELoss(reduction='sum')
        #classifier_loss = nn.CrossEntropyLoss()
        discriminator_loss = nn.BCELoss()
        self.zbuff = []
        self.zbuff_classes = []
        nsave_images=64

        accC_real_list = []
        accC_recon_list = []
        accC_gen_list = []
        accD_real_list = []
        accD_recon_list = []
        accD_gen_list = []
        reco_loss_list = []
        cls_loss_list = []
        disc_loss_list = []
        loss_list = []

        for e in range(start_from_epoch, self.hps.epochs_max):
            start = time.time()
            self.logr.start_epoch('Train', e)
            if self.G is not None: self.G.train() 
            if self.E is not None: self.E.train() 
            #self.Classifier.train()
            self.Discriminator.train()

            accC_real = 0
            accC_recon = 0
            accC_gen = 0
            accD_real = 0
            accD_recon = 0
            accD_gen = 0
            epoch_loss = 0

            for i, (x, target, xc) in enumerate(self.train_dataloader):
                # Get code directly from the dataset - bypass caching 
                batch_size = x.size(0)
                iter += batch_size

                if self.hps.use_cuda:
                    x = x.cuda()
                    xc = xc.cuda()

                # ENCODE
                if self.E is not None:
                    mu, varlog, ze, _, err_quant = self.E(xc)
                    z = self.reparam_log(mu, varlog) if self.hps.vae else mu
                    self.zbuff.append(ze.view(z.size(0), -1).detach().cpu().numpy())
                    labels_onehot = torch.zeros(target.shape[0], 10).to('cuda')
                    labels_onehot.scatter_(1, target.view(-1, 1).to('cuda'), 1)
                    labels_onehot = (labels_onehot * 2) - 1
                    target = target.view(target.size(0), -1)
                    self.zbuff_classes.append(target)

                    if self.hps.shared_weights:
                        self.ws = self.E.layers

                # DECODE
                if self.G is not None:
                    z = torch.cat((z, labels_onehot), dim=1)
                    xr = self.G(z, self.ws)

                # Calculate VAE loss
                log_dic = {}
                log_dic.update({'QERR': float(err_quant)})

                if self.G is not None and self.E is not None:
                    #xr = xr.view(xr.size(0), -1)
                    #x = x.view(x.size(0), -1)
                    if self.hps.binary_reco_loss:
                        loss_reco = torch.nn.functional.binary_cross_entropy(xr.view(x.size(0), -1), x.view(x.size(0), -1), reduction='none').sum() /batch_size
                    else:
                        pass
                        #loss_reco = mse(xr.view(xr.size(0), -1), x.view(x.size(0), -1)) / batch_size

                    if self.hps.vae:
                        varlog = torch.clamp(varlog, -10, 10) 
                        mu = torch.clamp(mu, -10, 10) 
                        loss_kld = -0.5 * torch.sum(1 + varlog - mu.pow(2) - varlog.exp()) 
                        loss_kld = loss_kld/batch_size
                        loss_kld = self.hps.kl_weight * loss_kld/self.hps.zsize
                        log_dic.update({'kld_loss': float(loss_kld)})
                        loss = loss + loss_kld

                    """
                    ## Update Classifier network
                    # Do remember to use .detach() here! - www.programmersought.com/article/152552205/
                    # Train with real images
                    self.optim_C.zero_grad()
                    y_cls = [ x[0] for x in target.tolist() ]
                    y_cls = torch.Tensor(y_cls).to('cuda').long()

                    prediction = self.Classifier(xc)
                    errC_real = classifier_loss(prediction, y_cls)
                    prediction = torch.argmax(prediction, 1).cpu().numpy()
                    accC_real += accuracy_score(prediction, y_cls.cpu().detach().numpy())

                    # Train with reconstructed images
                    prediction = self.Classifier(xr.detach())
                    errC_recon = classifier_loss(prediction, y_cls)
                    prediction = torch.argmax(prediction, 1).cpu().numpy()
                    accC_recon += accuracy_score(prediction, y_cls.cpu().detach().numpy())
                    """

                    S = [np.vstack(self.zbuff), np.vstack(self.zbuff_classes)]
                    x_g, y_g = self.eval(at_epoch=e,
                                         results_filename='eval_out.txt',
                                         latents=S)

                    """
                    # Train with generated images
                    prediction = self.Classifier(x_g.detach())
                    errC_gen = classifier_loss(prediction, y_g.long())
                    prediction = torch.argmax(prediction, 1).cpu().numpy()
                    accC_gen += accuracy_score(prediction, y_g.cpu().detach().numpy())

                    errC = (errC_real * 1) + (errC_recon * .5) + (errC_gen * .5)
                    errC.backward()
                    self.optim_C.step()
                    """

                    ## Update Discriminator network
                    # Do remember to use .detach() here! - www.programmersought.com/article/152552205/
                    # Train with real images
                    self.optim_D.zero_grad()
                    """
                    prediction = self.Discriminator(xc).view(-1)
                    yD_real = torch.ones((xc.shape[0])).to('cuda').float()
                    errD_real = discriminator_loss(prediction, yD_real)
                    accD_real += accuracy_score((prediction > 0.5).cpu().detach().numpy(),
                                                yD_real.cpu().detach().numpy())
                    """

                    # Train with reconstructed images
                    prediction = self.Discriminator(xr.detach()).view(-1)
                    #yD_fake = torch.zeros((xc.shape[0])).to('cuda').float()
                    yD_recon = torch.ones((xr.shape[0])).to('cuda').float()
                    errD_recon = discriminator_loss(prediction, yD_recon)
                    accD_recon += accuracy_score((prediction > 0.5).cpu().detach().numpy(),
                                                 yD_recon.cpu().detach().numpy())

                    # Train with generated images
                    prediction = self.Discriminator(x_g.detach()).view(-1)
                    yD_fake = torch.zeros((x_g.shape[0])).to('cuda').float()
                    errD_gen = discriminator_loss(prediction, yD_fake)
                    accD_gen += accuracy_score((prediction > 0.5).cpu().detach().numpy(),
                                               yD_fake.cpu().detach().numpy())

                    #errD_train = errD_real + (errD_recon * .5) + (errD_gen * .5)
                    errD_train = (errD_recon * 1) + (errD_gen * 1)
                    errD_train.backward()
                    self.optim_D.step()

                    ## Update Autoencoder network
                    # Do remember to NOT use .detach() here! - www.programmersought.com/article/152552205/
                    self.optim.zero_grad() 
                    """
                    # Run Classifier with real images
                    y_cls = [ x[0] for x in target.tolist() ]
                    y_cls = torch.Tensor(y_cls).to('cuda').long()
                    prediction = self.Classifier(xc)
                    errC_real = classifier_loss(prediction, y_cls)

                    # Run Classifier with reconstructed images
                    prediction = self.Classifier(xr)
                    errC_recon = classifier_loss(prediction, y_cls)

                    # Run Classifier with generated images
                    prediction = self.Classifier(x_g)
                    errC_gen = classifier_loss(prediction, y_g.long())

                    errC = (errC_real * 1) + (errC_recon * .5) + (errC_gen * .5)
                    """

                    # Run Discriminator with reconstructed images and real labels
                    """
                    prediction = self.Discriminator(xr).view(-1)
                    errD_recon = discriminator_loss(prediction, yD_real)
                    """

                    # Run Discriminator with generated images and real labels
                    prediction = self.Discriminator(x_g).view(-1)
                    errD_gen = discriminator_loss(prediction, yD_recon)

                    #errD = (errD_recon * .5) + (errD_gen * 1)
                    errD = errD_gen

                    loss_reco = mse(xr.view(xr.size(0), -1), x.view(x.size(0), -1)) / batch_size
                    #loss = (loss_reco * 1) + (errC * 1) + (errD * 1)
                    loss = (loss_reco * 1) + (errD * 1)

                    loss.backward()
                    self.optim.step()



                if self.G is not None: self.G.train() 
                if self.E is not None: self.E.train() 
                # LOGGING
                #=====================================
                if self.G is not None and self.E is not None:
                    log_dic.update({'loss': float(loss),
                                    'reco_loss':float(loss_reco),
                                    #'cls_loss':float(errC),
                                    'disc_loss':float(errD_train),
                                    #'accC_real':float(accC_real),
                                    #'accC_recon':float(accC_recon),
                                    #'accC_gen':float(accC_gen),
                                    #'accD_real':float(accD_real),
                                    'accD_recon':float(accD_recon),
                                    'accD_gen':float(accD_gen)})

                self.logr.log_loss(e, iter, stage_name='Train', losses=log_dic)

                if i % self.hps.print_every_batch == 0:
                    self.logr.print_batch_stat(stage_name='Train')

            #print(f'Classifier Accuracy: {}')
            accC_real /= len(self.train_dataloader)
            accC_recon /= len(self.train_dataloader)
            accC_gen /= len(self.train_dataloader)

            accC_real_list.append(accC_real)
            accC_recon_list.append(accC_recon)
            accC_gen_list.append(accC_gen)

            accD_real /= len(self.train_dataloader)
            accD_recon /= len(self.train_dataloader)
            accD_gen /= len(self.train_dataloader)

            accD_real_list.append(accD_real)
            accD_recon_list.append(accD_recon)
            accD_gen_list.append(accD_gen)

            reco_loss_list.append(loss_reco.item())
            #cls_loss_list.append(errC.item())
            disc_loss_list.append(errD_train.item())
            loss_list.append(loss.item())

            print(f'Epoch: {e} - Time: {time.time() - start}')
            self.logr.print_batch_stat('Train')


            # Record last batch of reconstructed images
            #======================================================================
            x = x[:nsave_images]
            xr = xr[:nsave_images]
            xc = xc[:nsave_images]

            size = list(x.size())
            if len(self.hps.corrupt_args) >0:
                size[0] = size[0]*3
                reco_imgs = torch.stack([x, xc.view(x.size(0), -1), xr], dim=1).view(size)
                cols = int(size[0]**0.5//3)*3
            else:
                size[0] = size[0]*2
                reco_imgs = torch.stack([x, xr], dim=1).view(size)
                cols = int(size[0]**0.5//2)*2

            self.logr.log_images(reco_imgs.cpu().detach(), e, 0, 'reconstructed_train', self.hps.channels, nrow=cols) 

            loss_reco_avg  = None
            if self.test_dataloader is not None:
                loss_reco_avg  = self.eval_reconstruct(self.test_dataloader, at_epoch=e, iter=iter)

            S = [np.vstack(self.zbuff), np.vstack(self.zbuff_classes)]
            self.zbuff = []
            self.zbuff_classes = []
            print("SAVING latents...", end='')
            pickle.dump(S, open(self.logr.exp_path+'/latents-last.pk', 'wb'))
            print('done')

            self.eval(at_epoch=e, results_filename='eval_out.txt')

            if not os.path.isfile(self.logr.exp_path+'/latents.pk'):
                print("latents.pk NOT found. Reseting best eval loss")
                self.current_best=9999

            if loss_reco_avg is not None and loss_reco_avg < self.current_best: 
                print("SAVING Best latents...", end='')
                print("loss_reco_avg < self.current_best", loss_reco_avg, self.current_best)
                pickle.dump(S, open(self.logr.exp_path+'/latents.pk', 'wb'))
                print('done')

            self.save_checkpoint(epoch=e, iter=iter, current_loss= loss_reco_avg)

        """
        fig = plt.figure()
        plt.plot(range(len(accC_real_list)), accC_real_list, label='Acc Real Data')
        plt.plot(range(len(accC_recon_list)), accC_recon_list, label='Acc Reconstructed Data')
        plt.plot(range(len(accC_gen_list)), accC_gen_list, label='Acc Generated Data')
        plt.ylim(0, 1.05)
        plt.title('Classifier')
        plt.legend()
        plt.savefig('cls_acc.png', dpi=300)
        """

        fig = plt.figure()
        #plt.plot(range(len(accD_real_list)), accD_real_list, label='Acc Real Data')
        plt.plot(range(len(accD_recon_list)), accD_recon_list, label='Acc Reconstructed Data')
        plt.plot(range(len(accD_gen_list)), accD_gen_list, label='Acc Generated Data')
        plt.ylim(0, 1.05)
        plt.title('Discriminator')
        plt.legend()
        plt.savefig('disc_acc.png', dpi=300)

        fig = plt.figure()
        plt.plot(range(len(reco_loss_list)), reco_loss_list, label='Reconstruction Loss')
        plt.legend()
        plt.savefig('recon_loss.png', dpi=300)

        """
        fig = plt.figure()
        plt.plot(range(len(cls_loss_list)), cls_loss_list, label='Classifier Loss')
        plt.legend()
        plt.savefig('cls_loss.png', dpi=300)
        """

        fig = plt.figure()
        plt.plot(range(len(disc_loss_list)), disc_loss_list, label='Discriminator Loss')
        plt.legend()
        plt.savefig('disc_loss.png', dpi=300)

        fig = plt.figure()
        plt.plot(range(len(loss_list)), loss_list, label='Total Loss')
        plt.legend()
        plt.savefig('total_loss.png', dpi=300)

        return

    def eval(self, at_epoch=0, results_filename=None, latents=None): 
        nsave_images=64
        imgs_per_row=8

        if self.G is None:
            return

        if self.hps.use_cuda:
            if self.G is not None and self.E is not None:
                self.G.cuda()
                self.E.cuda()

        if self.G is not None: self.G.eval()
        if self.E is not None: self.E.eval()
        torch.set_grad_enabled(False)


        #print("Generating samples("+self.hps.sample_method+"). N=",int(self.hps.gen_imgs), flush=True)

        imgs_reco_dir = os.path.join(self.logr.exp_path, 'generated', 'samples_'+self.hps.sample_method)
        #os.system('rm '+imgs_reco_dir+'/*.jpg')

        if self.hps.sample_method in ['cov', 'int']:
            latents_file=self.logr.exp_path+'/latents-last.pk'

            #print('Loading latents from:',latents_file)
            labels = []

            if latents:
                d = latents
            else:
                d = pickle.load(open(latents_file, 'rb'))

            if isinstance(d, list) and len(d) == 2:
                d, labels = d

            D = d 
            if len(d) == 2:
                # Trim labels
                d,D = d[0],D[0]

            L,mu,H=get_covb(d)

        elif self.hps.sample_method == 'random':
            pass
        else:
            print("Wrong sample method (",self.hps.sample_method,") only cov,int and random are implemented")
            sys.exit(0)

        # Generate samples
        zr=None
        for i in range(int(self.hps.gen_imgs)):

            # Calculate covariance of the real-img latents
            z_static = None
            test_batch_size = int(self.hps.batch_size_test)
            if self.hps.sample_method=='cov':
                z = sample(L, mu, test_batch_size, neg_zero=True, zsize=self.hps.zsize, ref=zr ,attr=None)
                z_static = torch.from_numpy(z.astype(np.float32))

            if self.hps.sample_method == 'int':
                steps = int(self.hps.interpolate_steps)

                imgs_per_row = steps+2
                z,hdists = interpolate_rnd(D, L, B=test_batch_size, steps=steps, 
                        labels=labels, set_attr = self.hps.set_attr)

                nsave_images=z.shape[0]
                z_static = torch.from_numpy(z.astype(np.float32))

            if z_static is None or self.hps.sample_method == 'random':
                if self.hps.vae:
                    z_static = torch.randn(test_batch_size, self.hps.zsize)
                else:
                    zt = z_static
                    z_static = torch.zeros([test_batch_size, self.hps.zsize]).uniform_(-1, 1)
                    z_static = torch.clamp(z_static, min=self.hps.zclamp_min, max=self.hps.zclamp)

                    z_static = roundf(z_static, self.hps.zround)
                    if zt is not None:
                        z_static = torch.cat([z_static, zt])

            if self.hps.use_cuda:
                z_static = z_static.cuda()

            labels = np.random.randint(2, size=z_static.shape[0])
            labels = torch.tensor(labels).long().to('cuda')
            labels_onehot = torch.zeros(labels.shape[0], 10).to('cuda')
            labels_onehot.scatter_(1, labels.view(-1, 1).to('cuda'), 1)
            labels_onehot = (labels_onehot * 2) - 1
            z_static = torch.cat((z_static, labels_onehot), dim=1)
            xr = self.G(z_static, None)

            tosave = int(self.hps.gen_imgs - (i)*xr.size(0))
            #save_images(xr[:tosave], self.hps.channels, self.hps.img_size, imgs_reco_dir, i*xr.size(0))

            name_suffix = self.hps.sample_method
            filename_img = self.logr.exp_path+'/generated/sample_'+str(at_epoch+i)+'_'+self.hps.sample_method
            if self.hps.set_attr > -1:
                filename_img += '_attr_'+str(self.hps.set_attr)
                name_suffix += '_attr_'+str(self.hps.set_attr)

            filename_img += '.pk'

            xrp = xr[:nsave_images]
            if latents is None:
                self.logr.log_images(xrp.cpu().detach(), at_epoch+i, 
                        name_suffix, 'generated', 
                        self.hps.channels, nrow=imgs_per_row) 

            if tosave <= xr.size(0):
                break

        torch.set_grad_enabled(True)

        # Store results
        if results_filename is not None:
            with open(results_filename, 'wt') as f:
                f.write('e:'+str(self.hps.epoch_start)+' loss_eval:'+str(self.current_best))

        return xr, labels

    def eval_reconstruct(self, dataset=None, at_epoch=0, iter=0, results_filename=None): 
        nsave_images=64
        e = at_epoch
        mse = nn.MSELoss(reduction='sum')

        if self.G is None:
            return

        if self.hps.use_cuda:
            if self.G is not None and self.E is not None:
                self.G.cuda()
                self.E.cuda()

        if self.G is not None: self.G.eval()
        if self.E is not None: self.E.eval()
        torch.set_grad_enabled(False)

        dir_suffix = 'eval'
        if self.hps.eval_train:
            dir_suffix = 'train'

        log_dic={}
        self.eval_latents=[]
        self.eval_classes=[]
        rnd_batch_imgs = None
        loss_total = 0
        samples = 0
        from_id = 0
        save_img_id = np.random.randint(1, len(dataset.dataset)//self.hps.batch_size_test-1)

        print("Evaluating samples. N=",len(dataset.dataset),  flush=False)
        self.logr.start_epoch('Eval', e)
        for i, (x, target, xc) in enumerate(dataset):
            target = target.view(target.size(0), -1)
            batch_size = x.size(0)

            if self.hps.use_cuda:
                x = x.cuda()
                xc = xc.cuda()

            varlog=mu=None
            loss_kld = 0

            if self.E is not None:
                mu, varlog, ze, _, err_quant = self.E(x)
                z = self.reparam_log(mu, varlog) if self.hps.vae else mu
                labels_onehot = torch.zeros(target.shape[0], 10).to('cuda')
                labels_onehot.scatter_(1, target.view(-1, 1).to('cuda'), 1)
                labels_onehot = (labels_onehot * 2) - 1
                self.eval_latents.append(z.view(z.size(0), -1).detach().cpu().numpy())
                self.eval_classes.append(target)

                self.ws = None
                if self.hps.shared_weights:
                    self.ws = self.E.layers

            if self.G is not None:
                z = torch.cat((z, labels_onehot), dim=1)
                xr = self.G(z, self.ws)

                # Calculate loss
                xr = xr.view(xr.size(0), -1)
                x = x.view(x.size(0), -1)

                if rnd_batch_imgs is None:
                    if save_img_id == i:
                        rnd_batch_imgs = [x, xr]

                if self.hps.binary_reco_loss:
                    loss_reco = torch.nn.functional.binary_cross_entropy(xr, x, reduction='sum')/batch_size
                else:
                    loss_reco = mse(xr, x) / batch_size

                if varlog is not None:
                    # varlog = torch.clamp(varlog, -10, 10) 
                    # mu = torch.clamp(mu, -10, 10) 
                    loss_kld = -0.5 * torch.sum(1 + varlog - mu.pow(2) - varlog.exp()) 
                    loss_kld = loss_kld/batch_size
                    loss_kld = self.hps.kl_weight * loss_kld/self.hps.zsize
                    log_dic.update({'kld_loss': float(loss_kld)})

                loss_total += float(loss_reco)

                samples +=1 
                log_dic.update({'loss': float(loss_kld+loss_reco), 'reco_loss':float(loss_reco ) })

            # Record the last loss etc. This is wrong it should be average, but for now it's ok
            # Save image and write out results
            imgs_test_dir = self.logr.exp_path+'/../'+self.hps.dataset+'_'+dir_suffix+'/'
            if self.hps.img_crop_size is not None:
                imgs_test_dir = self.logr.exp_path+'/../'+self.hps.dataset+'_'+str(self.hps.img_crop_size)+'_'+dir_suffix+'/'

            save_eval_imgs = True
            if os.path.isdir(imgs_test_dir): 
                image_list = glob.glob(os.path.join(imgs_test_dir, '*.jpg'))

                # Save the images only if they don't exist yet
                if len(image_list) == len(dataset.dataset): 
                    save_eval_imgs = False

            if save_eval_imgs:
                save_images(x, self.hps.channels, self.hps.img_size, imgs_test_dir, from_id)

            if self.hps.eval and not self.hps.eval_train:
                # Save all results
                ipath = os.path.join(self.logr.exp_path, 'reco')
                save_images(xr, self.hps.channels, self.hps.img_size, ipath, from_id)

            iter += batch_size
            from_id += batch_size

        # Record the last loss etc. This is wrong it should be average, but for now it's ok
        loss_reco_avg = float(loss_total / samples)
        log_dic.update({'loss': float(0), 'reco_loss':loss_reco_avg })
        self.logr.log_loss(e, None, stage_name='Eval', losses=log_dic)

        S = [np.vstack(self.eval_latents), np.vstack(self.eval_classes)]
        pickle.dump(S, open(self.logr.exp_path+'/latents-'+dir_suffix+'.pk', 'wb'))

        # Save random batch of images or the last one 
        if rnd_batch_imgs is not None:
            x,xr = rnd_batch_imgs

        # Save only the first nsave_images images
        x = x[:nsave_images]
        xr = xr[:nsave_images]
            
        size = list(x.size())
        size[0] = size[0]*2
        reco_imgs = torch.stack([x, xr], dim=1).view(size)
        cols = int(size[0]**0.5//2)*2
        self.logr.log_images(reco_imgs.cpu().detach(), e, 0, 'reconstructed_'+dir_suffix, self.hps.channels, nrow=cols) 


        self.logr.print_batch_stat('Eval')
        torch.set_grad_enabled(True)
        print('')

        if results_filename is not None:
            with open(results_filename, 'wt') as f:
                f.write('e:'+str(self.hps.epoch_start)+' loss_eval:'+str(self.current_best))

        return loss_reco_avg


def exec(hps):
    hps.eval = False
    hps.gen = False
    hps.reload=False
    hps.load_from_sys_args(sys.argv)


    logr = MLLogger(hps)
    experiment_name=hps.cfg+ '_res'+str(hps.img_size)+\
                (('_'+hps.vae_model) if hps.vae_model is not None else '')+\
                ('-vae' if hps.vae else '-qae')+\
                '-z'+str(hps.zsize)+'-'+hps.exp_suffix

    # Check whether we are in the experiment directory rather than the root dir
    exp_path = os.path.split(os.getcwd())[-1]
    if exp_path == experiment_name:
        experiment_name = '.'

    # Open existing or create a new expriment
    logr.open_experiment(experiment_name)

    # Print system info and configuration parameters
    print(' '.join(sys.argv))
    # print_pkg_versions()
    print(hps)

    sv = Solver(hps, logr)
    sv.net_init()
    sv.load_data()

    # Load model if specified
    if hps.reload:
        filename = logr.model_path+'/weights-'+str(int(hps.reload))+'.cp' if not isinstance(hps.reload, bool) else None
        hps.epoch_start = sv.load_checkpoint(filename) 
    if hps.l:
        hps.epoch_start = sv.load_checkpoint(filename=logr.model_path+'/last.cp') 

    if hps.eval: 
        dataloader = sv.test_dataloader
        results_filename = logr.exp_path +'/fid-epoch.txt'

        if not hps.l and not hps.reload:
            hps.epoch_start = sv.load_checkpoint(filename=logr.model_path+'/last.cp') 

        if hps.gen:
            # Novel imgs generation, interpolation, attributes modification
            sv.eval(results_filename=results_filename)
        else:
            # Reconstruction evaluation
            if hps.eval_train: 
                # On the train dataset
                sv.eval_reconstruct(dataset=sv.train_dataloader)
            else:
                # On the test dataset
                sv.eval_reconstruct(dataset=dataloader, results_filename=results_filename)
    else:
        sv.train()
    return

#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")

