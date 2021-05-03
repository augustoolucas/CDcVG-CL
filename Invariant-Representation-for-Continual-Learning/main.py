# Author: Ghada Sokar et al.
# This is the implementation for the Learning Invariant Representation for Continual Learning paper in AAAI workshop on Meta-Learning for Computer Vision
# if you use part of this code, please cite the following article:
# @inproceedings{sokar2021learning,
#       title={Learning Invariant Representation for Continual Learning}, 
#       author={Ghada Sokar and Decebal Constantin Mocanu and Mykola Pechenizkiy},
#       booktitle={Meta-Learning for Computer Vision Workshop at the 35th AAAI Conference on Artificial Intelligence (AAAI-21)},
#       year={2021},
# }  

import argparse
import glob
import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import optuna
import mlflow
import joblib
#from apex import amp
import torch
from tqdm import tqdm
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
#from torch.autograd import Variable

from model import *
import data_utils
import plot_utils

def parse_args():
    desc = "Pytorch implementation of Learning Invariant Representation for CL (IRCL) on the Split MNIST benchmark"
    parser = argparse.ArgumentParser(description=desc)

    # data
    parser.add_argument('--cvae_model', type=str, default='mlp', help='Autoencoder Model: MLP, MLP-Enlarged or ConvNet')
    parser.add_argument('--specific_model', type=str, default='mlp', help='Autoencoder Model: MLP, MLP-Enlarged or ConvNet')
    parser.add_argument("--img_size", type=int, default=28, help="dimensionality of the input image")
    parser.add_argument("--channels", type=int, default=1, help="dimensionality of the input channels")
    parser.add_argument("--n_classes", type=int, default=10, help="total number of classes")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Options: MNIST, EMNIST, Fashion-MNIST, CIFAR10, SVHN")

    # architecture
    parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent code")
    parser.add_argument('--n_hidden_cvae', type=int, default=300, help='number of hidden units in conditional variational autoencoder')
    parser.add_argument('--n_hidden_specific', type=int, default=20, help='number of hidden units in the specific module')
    parser.add_argument('--n_hidden_classifier', type=int, default=40, help='number of hidden units in the classification module')
    
    # training parameters
    parser.add_argument('--learn_rate', type=float, default=1e-2, help='learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='the number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='test Batch size')
    parser.add_argument("--log_interval", type=int, default=50, help="interval between logging")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--seed", type=int, default=1, help="seed") 

    # visualization
    parser.add_argument('--results_path', type=str, default='results',
                        help='the path of output images (generated and reconstructed)')
    parser.add_argument('--n_img_x', type=int, default=8,
                        help='number of images along x-axis')
    parser.add_argument('--n_img_y', type=int, default=8,
                        help='number of images along y-axis')


    return check_args(parser.parse_args())


def check_args(args):
    # results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)
    return args


def check_path(path):
    try:
        os.mkdir(path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)


args = parse_args()
if args is None:
    exit()

def save_train_imgs(original, reconstructed, samples, task_id, path):
    cmap = 'gray' if args.channels == 1 else None
    original = original.astype(np.uint8)
    reconstructed = reconstructed.astype(np.uint8)
    samples = samples.astype(np.uint8)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(original, cmap=cmap)
    ax1.set_xlabel('Original')
    ax2.imshow(reconstructed, cmap=cmap)
    ax2.set_xlabel('Reconstructed')
    ax3.imshow(samples, cmap=cmap)
    ax3.set_xlabel('Generated')

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)

    plt.savefig(dpi=300,
                fname='%s/imgs_task_%d.jpg' % (path, task_id),
                bbox_inches='tight')
    plt.close()


def save_train_losses(num_epochs, losses, task_id, path):
    fig = plt.figure(figsize=(20, 8))#, tight_layout={'pad':0}, frameon=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for idx, (k, v) in enumerate(losses.items()):
        if idx != len(losses) - 1:
            ax = plt.subplot2grid(shape=(2, 4), loc=(idx%2, idx//2)) 
        else:
            ax = plt.subplot2grid(shape=(7, 4), loc=(2, idx//2), rowspan=3)
        ax.plot(list(range(num_epochs)), v)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Epochs')

        if k == 'cvae':
            ylabel = f'{k.upper()} Loss (Rec + KL)'
        elif k == 'rec':
            ylabel = f'{k.capitalize()} Loss'
        elif k == 'kl':
            ylabel = f'{k.upper()} Loss'
        elif k == 'classifier':
            ylabel = f'{k.capitalize()} Loss'
        elif k == 'total':
            ylabel = f'{k.capitalize()} Loss'

        ax.set_ylabel(ylabel)

    plt.savefig(dpi=300,
                fname='%s/losses_task_%d.jpg' % (path, task_id),
                bbox_inches='tight')
    

def visualize(args, test_loader, encoder, decoder, latent_dim, img_shape, n_classes, curr_task_labels, device, task_id, path):
    plotter = plot_utils.plot_samples(path, img_shape, args.n_img_x, args.n_img_y)
    # plot samples of the reconstructed images from the first batch of the test set of the current task
    for test_batch_idx, (test_data, test_target) in enumerate(test_loader):  
        test_data, test_target = test_data.to(device), test_target.to(device)                  
        x = test_data[0:plotter.n_total_imgs, :]
        x_id = test_target[0:plotter.n_total_imgs]
        x_id_onehot = get_categorical(x_id,n_classes).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z,_,_ = encoder(x)
            z = torch.cat((z, x_id_onehot), dim=1)
            reconstructed_x = decoder(z)

            x_img = plotter.get_images(x.cpu().data)
            reconstructed_x_img = plotter.get_images(reconstructed_x.cpu().data)
        break
    
    #plot pseudo random samples from the previous learned tasks
    z = Variable(Tensor(np.random.normal(0, 1, (plotter.n_total_imgs, latent_dim))))
    z_id = np.random.randint(0, curr_task_labels[-1]+1, size=[plotter.n_total_imgs])  
    z_id = torch.tensor(z_id, dtype=torch.int64)
    z_id_onehot = get_categorical(z_id, n_classes).to(device)
    decoder.eval()
    with torch.no_grad():
        z = torch.cat((z, z_id_onehot), dim=1)
        pseudo_samples = decoder(z)
        pseudo_img = plotter.get_images(pseudo_samples.cpu().data)

    save_train_imgs(x_img, reconstructed_x_img, pseudo_img, task_id, path)


def get_categorical(labels, n_classes):
    if args.cvae_model == 'convnet':
        labels_onehot = torch.zeros(labels.shape[0], n_classes).to(device)
        labels_onehot.scatter_(1, labels.view(-1, 1).to(device), 1)
    elif args.cvae_model == 'mlp':
        labels_onehot = np.array(labels.data.tolist())
        labels_onehot = np.eye(n_classes)[labels_onehot].astype('float32')
        labels_onehot = torch.from_numpy(labels_onehot)

    return Variable(labels_onehot)

def generate_pseudo_samples(device, task_id, latent_dim, curr_task_labels, decoder, replay_count, n_classes):
    gen_count = sum(replay_count[0:task_id])
    z = Variable(Tensor(np.random.normal(0, 1, (gen_count, latent_dim))))    
    # this can be used if we want to replay different number of samples for each task
    for i in range(task_id):
        if i==0:
            x_id_ = np.random.randint(0, curr_task_labels[i][-1]+1, size=[replay_count[i]])  
        else:
            x_id_ = np.concatenate((x_id_,np.random.randint(curr_task_labels[i][0], curr_task_labels[i][-1]+1, size=[replay_count[i]])))

    np.random.shuffle(x_id_)
    x_id_ = torch.tensor(x_id_, dtype=torch.int64, device=device)
    x_id_one_hot = get_categorical(x_id_, n_classes).to(device)
    decoder.eval()
    with torch.no_grad():
        z = torch.cat((z, x_id_one_hot), dim=1)
        x = decoder(z)
    return x, x_id_

def evaluate(encoder, classifier, task_id, device, task_test_loader, path, final_tests=False):
    correct_class = 0    
    n = 0
    classifier.eval()
    encoder.eval()
    with torch.no_grad():
        for data, target in task_test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            n += target.shape[0]
            z_representation,_,_ = encoder(data)
            model_output = classifier(data, z_representation)
            pred_class = model_output.argmax(dim=1, keepdim=True)
            correct_class += pred_class.eq(target.view_as(pred_class)).sum().item()

    if final_tests:
        with open(f'{path}/log.txt', 'a+') as writer:
            print('Test evaluation of task_id: {} ACC: {}/{} ({:.3f}%)'.format(
            task_id, correct_class, n, 100*correct_class/float(n)), file=writer)

    print('Test evaluation of task_id: {} ACC: {}/{} ({:.3f}%)'.format(
         task_id, correct_class, n, 100*correct_class/float(n)))  

    if not final_tests:
        if 100*correct_class/float(n) < 80:
            print('Task accuracy too low, prunning trial.')
            raise optuna.exceptions.TrialPruned()

    return 100. * correct_class / float(n)


def train(num_epochs, n_classes, latent_dim, batch_size, optimizer_cvae, optimizer_C, encoder, decoder,classifer, img_shape, train_loader, test_loader, curr_task_labels, task_id, device, path):
    ## loss ##
    pixelwise_loss = nn.MSELoss(reduction='sum')
    classification_loss = nn.CrossEntropyLoss()
    encoder.train()
    decoder.train()
    classifer.train()
    losses_over_time = {'cvae': [],
                        'rec': [],
                        'kl': [],
                        'classifier': [],
                        'total': []}

    cvae_loss_over_epochs = []
    rec_loss_over_epochs = []
    kl_loss_over_epochs = []
    classifier_loss_over_epochs = []
    total_loss_over_epochs = []
    for epoch in tqdm(range(num_epochs)):
        #print(f'Epoch: {epoch}')
        losses = {'cvae': 0,
                  'rec': 0,
                  'kl': 0,
                  'classifier': 0,
                  'total': 0}
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            #---------------------------#
            ## train encoder-decoder ##
            #---------------------------#
            encoder.zero_grad()
            decoder.zero_grad()
            classifer.zero_grad()

            y_onehot = get_categorical(target, n_classes).to(device)
            encoded_imgs,z_mu,z_var = encoder(data)
            ### Add condition
            encoded = torch.cat((encoded_imgs, y_onehot), dim=1)
            decoded_imgs = decoder(encoded)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)/batch_size
            rec_loss = pixelwise_loss(decoded_imgs, data)/batch_size
            cvae_loss = rec_loss + kl_loss

            #with amp.scale_loss(cvae_loss, optimizer_cvae) as cvae_loss:
            cvae_loss.backward()

            optimizer_cvae.step()

            #---------------------------#
            ## train Classifer ##
            #---------------------------#
            encoder.zero_grad()
            decoder.zero_grad()
            classifer.zero_grad()
            z_representation,_,_ = encoder(data)
            # the classifier includes the specific module
            outputs = classifer(data, z_representation.detach())
            c_loss = classification_loss(outputs, target)

            #with amp.scale_loss(c_loss, optimizer_C) as c_loss:
            c_loss.backward()

            optimizer_C.step()

            total_loss = cvae_loss.item() + c_loss.item()
            
            if batch_idx % args.log_interval == 0:
                losses['cvae'] = cvae_loss.item()
                losses['rec'] = rec_loss.item()
                losses['kl'] = kl_loss.item()
                losses['classifier'] = c_loss.item()
                losses['total'] = total_loss
        
        losses_over_time['cvae'].append(losses['cvae'])
        losses_over_time['rec'].append(losses['rec'])
        losses_over_time['kl'].append(losses['kl'])
        losses_over_time['classifier'].append(losses['classifier'])
        losses_over_time['total'].append(losses['total'])
        
        if epoch+1 == num_epochs:
           test_acc = evaluate(encoder, classifer, task_id, device, test_loader, path, final_tests=False)
           visualize(args, test_loader, encoder, decoder, latent_dim, img_shape, n_classes, curr_task_labels, device, task_id, path)
    
    save_train_losses(num_epochs, losses_over_time, task_id, path)

    return test_acc

def main(trial, experiment_path):
    # set seed
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("used device: " + str(device).upper())

    ## DATA ##
    # Load data and construct the tasks 
    train_dataset, test_dataset = data_utils.load_data(args.dataset)
    img_shape = tuple(train_dataset.data.shape[1:])

    if len(img_shape) == 2:
        img_shape = tuple([1] + list(img_shape))

    n_classes = len(set(train_dataset.targets.tolist()))
    task_labels = [[x, y] for x, y in zip(range(0, n_classes, 2), range(1, n_classes, 2))]
    if not n_classes % 2 == 0:
        task_labels.append([args.n_classes - 1])

    num_tasks = len(task_labels)
    num_replayed = [5000] * num_tasks
    train_dataset, test_dataset = data_utils.task_construction(task_labels, 
                                                               train_dataset,
                                                               test_dataset)
    
    latent_dim = trial.suggest_int('latent_dim', 16, 64, 16)

    specific_representation_size = trial.suggest_int('specific_representation_size',
                                                     10, 60, 10)

    n_layers_classifier = trial.suggest_int('num_layers_classifier',
                                                   1, 3, 1)
    layer_size_classifier = trial.suggest_int('max_layer_size_classifier',
                                                  20, 80, 10)
    results_path = f'{experiment_path}/Trial {trial.number}'
    check_path(results_path)
    ## MODEL ## 
    # Initialize encoder, decoder, specific, and classifier
    if args.cvae_model == 'mlp':
        n_layers_cvae = trial.suggest_int('num_layers_cvae', 1, 3, 1)
        layer_size_cvae = trial.suggest_int('max_layer_size_cvae', 300, 600, 100)

        encoder = Encoder(img_shape, layer_size_cvae, latent_dim, 
                          n_layers_cvae).to(device)
        decoder = Decoder(img_shape, layer_size_cvae, latent_dim,
                          n_classes, n_layers=n_layers_cvae).to(device)
    elif args.cvae_model == 'convnet':
        encoder = ConvEncoder(img_shape, latent_dim).to(device)
        decoder = ConvDecoder(img_shape, latent_dim + n_classes).to(device)
    else:
        print(f'Invalid cvae_model argument: {args.cvae_model}')
        exit()

    if args.specific_model == 'mlp':
        n_layers_specific = trial.suggest_int('num_layers_specific', 1, 3, 1)
        classifier = Classifier(img_shape, latent_dim,
                                specific_representation_size,
                                layer_size_classifier,
                                n_classes, n_layers_specific,
                                n_layers_classifier).to(device)
    elif args.specific_model == 'convnet':
        classifier = ConvClassifier(img_shape, specific_representation_size,
                                    n_layers_classifier,
                                    layer_size_classifier,
                                    n_classes, latent_dim).to(device)
    else:
        print(f'Invalid specific_model argument: {args.specific_model}')
        exit()
    
    cvae_lr = trial.suggest_loguniform('cvae_learning_rate', 1e-5, 1e-2)
    classifier_lr = trial.suggest_loguniform('classifier_learning_rate', 1e-5, 1e-2)
    ## OPTIMIZERS ##
    optimizer_cvae = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=cvae_lr)
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)

    """
    [encoder, decoder], optimizer_cvae = amp.initialize([encoder, decoder],
                                                        optimizer_cvae,
                                                        opt_level='O1',
                                                        verbosity=0)
    classifier, optimizer_C = amp.initialize(classifier,
                                             optimizer_C,
                                             opt_level='O1',
                                             verbosity=0)
    """

    test_loaders = []
    acc_of_task_t_at_time_t = [] # acc of each task at the end of learning it

    #------------------------------------------------------------------------------------------#
    #----- Train the sequence of CL tasks -----#
    #----------------------------------------------------------------------#

    train_batch_size = trial.suggest_int('train_batch_size', 64, 512, 64)
    num_epochs = trial.suggest_int('num_epochs', 5, 500, 5)

    print(trial.params)
    for task_id in range(num_tasks):
        torch.cuda.empty_cache()
        print("Strat training task#" + str(task_id))
        sys.stdout.flush()
        if task_id > 0:            
            # generate pseudo-samples of previous tasks
            gen_x,gen_y = generate_pseudo_samples(device, task_id, latent_dim, task_labels, decoder, num_replayed, args.n_classes)

            if gen_x.shape[1] == 3:
                gen_x = gen_x.reshape([gen_x.shape[0], img_shape[1],img_shape[2], img_shape[0]])
            else:
                gen_x = gen_x.reshape([gen_x.shape[0], img_shape[1],img_shape[2]])

            train_dataset[task_id-1].data = (gen_x*255).type(torch.uint8)
            train_dataset[task_id-1].targets = Variable(gen_y).type(torch.long)
            # concatenate the pseduo samples of previous tasks with the data of the current task
            
            if type(train_dataset[task_id-1].data) == torch.Tensor and type(train_dataset[task_id].data) == np.ndarray:
                train_dataset[task_id-1].data = train_dataset[task_id-1].data.to('cpu').numpy()
                train_dataset[task_id-1].targets = train_dataset[task_id-1].targets.to('cpu').numpy()
                
                train_dataset[task_id].data = np.concatenate((train_dataset[task_id].data,train_dataset[task_id-1].data))
                train_dataset[task_id].targets =  np.concatenate((train_dataset[task_id].targets, train_dataset[task_id-1].targets))
            else:
                train_dataset[task_id].data = torch.cat((train_dataset[task_id].data,train_dataset[task_id-1].data.cpu()))
                train_dataset[task_id].targets =  torch.cat((train_dataset[task_id].targets, train_dataset[task_id-1].targets.cpu()))

        train_loader = data_utils.get_train_loader(train_dataset[task_id], train_batch_size)
        test_loader = data_utils.get_test_loader(test_dataset[task_id], args.test_batch_size)
        test_loaders.append(test_loader)
        # train current task
        test_acc = train(num_epochs, n_classes, latent_dim, train_batch_size,
                         optimizer_cvae, optimizer_C, encoder, decoder,
                         classifier, img_shape, train_loader, test_loader,
                         task_labels[task_id], task_id, device, results_path)
        acc_of_task_t_at_time_t.append(test_acc)
        print('\n')
        sys.stdout.flush()
    #------------------------------------------------------------------------------------------#
    #----- Performance on each task after training the whole sequence -----#
    #----------------------------------------------------------------------#
    ACC = 0
    BWT = 0
    test_accs = []

    with open(f'{results_path}/log.txt', 'a+') as writer:
        print(f'Img Shape: {img_shape}', file=writer)
        print(f'Classes: {n_classes}', file=writer)
        print(f'Dataset: {args.dataset}', file=writer)
        print(f'Epochs: {args.num_epochs}', file=writer)
        print(f'CVAE Learning rate: {cvae_lr}', file=writer)
        print(f'Classifier Learning rate: {classifier_lr}', file=writer)
        print(f'Batch Size: {train_batch_size}', file=writer)

        if args.cvae_model == 'mlp':
            print(f'Hidden Units CVAE: {layer_size_cvae}', file=writer)

        if args.specific_model == 'mlp':
            print(f'Hidden Units Specific: {args.n_hidden_specific}', file=writer)

        print(f'Hidden Units Classifier: {args.n_hidden_classifier}', file=writer)
        print('', file=writer)

    for task_id in range(num_tasks):
        task_acc = evaluate(encoder, classifier, task_id, device,
                            test_loaders[task_id], results_path,
                            final_tests=True)
        test_accs.append(task_acc)
        ACC += task_acc
        BWT += (task_acc - acc_of_task_t_at_time_t[task_id])

    plt.figure()
    plt.plot(list(range(num_tasks)), test_accs, marker='o')
    plt.locator_params(axis='x', integer=True)
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.savefig(dpi=200, fname=f'{results_path}/acc_by_tasks.png')

    ACC = ACC/len(task_labels)
    BWT = BWT/len(task_labels)-1
    with open(f'{results_path}/log.txt', 'a+') as writer:
        print('Average accuracy in task agnostic inference (ACC):  {:.3f}'.format(ACC), file=writer)
        print('Average backward transfer (BWT): {:.3f}'.format(BWT), file=writer)

    print('Average accuracy in task agnostic inference (ACC):  {:.3f}'.format(ACC))
    print('Average backward transfer (BWT): {:.3f}'.format(BWT))

    with mlflow.start_run():
        mlflow.log_param('Trial', trial.number)
        for k, v in trial.params.items():
            mlflow.log_param(k, v)

        mlflow.log_param('Image_Shape', img_shape)
        mlflow.log_param('CVAE_Model', args.cvae_model)
        mlflow.log_param('Specific_Model', args.specific_model)

        for idx, v in enumerate(test_accs):
            mlflow.log_metric(f'Acc Task {idx}', v)

        mlflow.log_metric('Avg Acc', ACC)

    return ACC


if __name__ == '__main__':
    experiment_name = f'{args.cvae_model}_cvae_{args.specific_model}_specific'
    experiment_path = f'./Optuna/{experiment_name}'
    OPTUNA_DUMP = f'{experiment_name}.pkl'

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        
    if os.path.exists(f'{experiment_path}/{OPTUNA_DUMP}'):
        study = joblib.load(f'{experiment_path}/{OPTUNA_DUMP}')
        print("Best trial until now:")
        print(" Value: ", study.best_trial.value)
        print(" Params: ")

        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        study = optuna.create_study(study_name=experiment_name,
                                    direction='maximize')

    mlflow.set_experiment(experiment_name=experiment_name)
    study.optimize(lambda trial: main(trial, experiment_path), n_trials=25, n_jobs=1)
    joblib.dump(study, f'./{experiment_path}/{OPTUNA_DUMP}')
