import os
import yaml
import copy
import torch
import mlflow
import models
import random
import utils.data
import utils.plot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.manifold import TSNE
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

def gen_pseudo_samples(n_gen, tasks_labels, decoder, n_classes, latent_dim, device):
    decoder.eval()

    labels = [label for labels in tasks_labels for label in labels]
    for label in labels:
        y = torch.ones(size=[n_gen], dtype=torch.int64) * label
        new_labels = y if label == 0 else torch.cat((new_labels, y))

    new_labels = new_labels[torch.randperm(new_labels.size(0))] # Shuffling the Tensor
    new_labels_onehot = utils.data.onehot_encoder(new_labels, n_classes)

    z = torch.Tensor(np.random.normal(0, 1, (n_gen * len(labels), latent_dim)))

    with torch.no_grad():
        input = torch.cat([z, new_labels_onehot], dim=1)
        input = torch.split(input, 128)
        new_images = torch.Tensor([])

        for chunk in input:
            gen_images = decoder(chunk.to(device))
            gen_images = gen_images.to('cpu')
            new_images = torch.cat([new_images, gen_images])
        torch.cuda.empty_cache()

    return new_images, new_labels


def test(encoder, specific, classifier, data_loader, device, fname=None):
    encoder.eval(); classifier.eval(); specific.eval()

    batch_acc = 0
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for images, labels in data_loader:
            images = images.to(device)
            latents, _, _ = encoder(images)

            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())

            classifier_output = torch.argmax(classifier_output, dim=1).tolist()
            batch_acc += accuracy_score(classifier_output, labels.tolist())

            if fname is not None:
                all_outputs.extend(classifier_output)
                all_labels.extend(labels.tolist())

        batch_acc /= len(data_loader)

    if fname is not None:
        cm = confusion_matrix(all_labels, all_outputs, labels=list(set(all_labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=list(set(all_labels)))
        disp.plot(colorbar=False)
        plt.savefig(dpi=300, fname=fname, bbox_inches='tight')
        plt.close()

    return batch_acc 


def gen_recon_images(encoder, decoder, data_loader, device):
    encoder.eval(); decoder.eval()

    ### Generate reconstructed images ###
    with torch.no_grad():
        images, labels = next(iter(data_loader))
        labels_onehot = utils.data.onehot_encoder(labels, 10).to(device)
        images = images.to(device)
        latents, _, _ = encoder(images)
        images = images.to('cpu')
        recon_images = decoder(torch.cat([latents, labels_onehot], dim=1)).to('cpu')

    return images, recon_images


def train_task(config, encoder, decoder, specific, classifier, train_loader, val_loader, tasks_labels, task_id, device):
    ### ------ Optimizers ------ ###
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=float(config['lr_autoencoder']))
    optimizer_specific = torch.optim.Adam(specific.parameters(),
                                          lr=float(config['lr_specific']))
    optimizer_classifier = torch.optim.Adam(classifier.parameters(),
                                            lr=float(config['lr_classifier']))

    mlflow.log_params({'optimizer_cvae': 'Adam',
                       'optimizer_specific': 'Adam',
                       'optimizer_classifier': 'Adam'})

    ### ------ Losses ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    classification_loss = torch.nn.CrossEntropyLoss()
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    mlflow.log_params({'pixelwise_loss': 'MSE',
                       'classification_loss': 'CrossEntropyLoss'})
    ### ------ Training ------ ###

    task_plt_path = f'{config["plt_path"]}/Task_{task_id}'
    if not Path(task_plt_path).is_dir():
        os.mkdir(task_plt_path)

    train_bar = tqdm(range(config['epochs']))

    cvae_loss_epochs, rec_loss_epochs, kl_loss_epochs = [], [], []
    classifier_loss_epochs, total_loss_epochs = [], []
    for epoch in train_bar:
        epoch_classifier_loss, epoch_rec_loss, epoch_kl_loss = 0, 0, 0
        epoch_acc = 0
        classes = {}
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            for cls in set(labels.tolist()):
                classes[cls] = classes[cls] + len(labels[labels == cls]) if cls in classes else len(labels[labels == cls])
            encoder.train(); decoder.train(); specific.train(); classifier.train()
            ### ------ Training autoencoder ------ ###
            encoder.zero_grad(); decoder.zero_grad()
            specific.zero_grad(); classifier.zero_grad()

            labels_onehot = utils.data.onehot_encoder(labels, 10).to(device)
            images, labels = images.to(device), labels.to(device)
            latents, mu, var = encoder(images)
            recon_images = decoder(torch.cat([latents, labels_onehot], dim=1))

            kl_loss = kl_divergence_loss(var, mu)/config['batch_size']
            rec_loss = pixelwise_loss(recon_images, images)/config['batch_size']
            cvae_loss = rec_loss + kl_loss

            cvae_loss.backward()
            optimizer_cvae.step()

            ### ------ Training specific module and classifier ------ ###
            encoder.eval(); decoder.eval()
            encoder.zero_grad(); decoder.zero_grad()
            specific.zero_grad(); classifier.zero_grad()

            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())
            classifier_loss = classification_loss(classifier_output, labels)
            classifier_loss.backward()
            optimizer_classifier.step()
            optimizer_specific.step()

            output_list = torch.argmax(classifier_output, dim=1).tolist()
            epoch_acc += accuracy_score(output_list, labels.detach().cpu().numpy())

            epoch_classifier_loss += classifier_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()

        epoch_loss = epoch_classifier_loss + epoch_rec_loss + epoch_kl_loss
        mlflow.log_metrics({f'rec_loss_task_{task_id}': epoch_rec_loss/len(train_loader),
                            f'kl_loss_task_{task_id}': epoch_kl_loss/len(train_loader),
                            f'cvae_loss_task_{task_id}': (epoch_rec_loss + epoch_kl_loss)/len(train_loader),
                            f'classifier_loss_task_{task_id}': epoch_classifier_loss/len(train_loader),
                            f'total_loss_task_{task_id}': epoch_loss/len(train_loader)},
                          step=epoch)
        rec_loss_epochs.append(epoch_rec_loss/len(train_loader))
        kl_loss_epochs.append(epoch_kl_loss/len(train_loader))
        cvae_loss_epochs.append((epoch_rec_loss + epoch_kl_loss)/len(train_loader))
        classifier_loss_epochs.append(epoch_classifier_loss/len(train_loader))
        total_loss_epochs.append((epoch_loss/len(train_loader)))
        val_acc = test(encoder, specific, classifier, val_loader, device, f'{task_plt_path}/epoch_{epoch}')
        
        train_bar.set_description(f'Epoch: {(epoch + 1)}/{config["epochs"]} - '
                                  f'Loss: {(epoch_loss/len(train_loader)):.03f} - '
                                  f'Accuracy: {(epoch_acc/len(train_loader))*100:.03f}% - '
                                  f'Val Accuracy: {(val_acc)*100:.03f}%')

    real_images, recon_images = gen_recon_images(encoder, decoder, val_loader, device)
    gen_images, _ = gen_pseudo_samples(128, tasks_labels, decoder, 10, config['latent_size'], device)

    fig = plt.figure()
    plt.bar(list(classes.keys()), list(classes.values()))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, key in enumerate(classes):
        ax.annotate(classes[key],
                    xy=(idx-len(classes)*0.035, classes[key]+(max(classes.values()) - min(classes.values()))*0.025),
                    fontsize=10)

    plt.title(f'Task {task_id}')
    plt.savefig(dpi=300,
                fname=f'{task_plt_path}/classes_distribution.png',
                bbox_inches='tight')
    plt.close()

    utils.plot.plot_losses(classifier_loss_epochs,
                           xlabel='Epoch',
                           ylabel='Loss',
                           title=f'Task {task_id}',
                           fname=f'{task_plt_path}/loss-classifier.png')

    utils.plot.plot_losses(total_loss_epochs,
                           xlabel='Epoch',
                           ylabel='Loss',
                           title=f'Task {task_id}',
                           fname=f'{task_plt_path}/loss-total.png')

    utils.plot.plot_losses(cvae_loss_epochs,
                           xlabel='Epoch',
                           ylabel='Reconstruction + KL Loss',
                           title=f'Task {task_id}',
                           fname=f'{task_plt_path}/loss-cvae.png')

    utils.plot.multi_plots(data1=rec_loss_epochs,
                           data2=kl_loss_epochs,
                           xlabel='Epoch',
                           ylabel1='Reconstruction Loss',
                           ylabel2='KL Loss',
                           title=f'Task {task_id}',
                           fname=f'{task_plt_path}/loss-rec-kl.png')

    utils.plot.visualize(real_images, recon_images, gen_images, f'{task_plt_path}/images.png')


def sne(path, encoder, specific, classifier, knn, data_loader, device, data_type):
    encoder.eval(); classifier.eval(); specific.eval()

    with torch.no_grad():
        specific_list = []
        latents_list = []
        labels_list = []
        combined_list = []
        classifier_out_list = []
        for images, labels in data_loader:
            images = images.to(device)
            latents, _, _ = encoder(images)
            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())

            classifier_out_list.extend(torch.argmax(classifier_output, dim=1).cpu().tolist())
            specific_list.extend(specific_output.cpu().tolist())
            latents_list.extend(latents.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())
            combined_list.extend(torch.cat([latents, specific_output], dim=1).cpu().tolist())

    mlp_acc = 100*accuracy_score(labels_list, classifier_out_list)

    knn_output = knn.predict(combined_list)
    knn_acc = 100*accuracy_score(labels_list, knn_output)
    with open(f'{path}/output.log', 'a') as f:
        print(f'{data_type.capitalize()} KNN Accuracy: {knn_acc:.02f}%', file=f)
    print(f'{data_type.capitalize()} KNN Accuracy: {knn_acc:.02f}%')
    mlflow.log_metric(f'{data_type.capitalize()} KNN Accuracy', knn_acc)

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, init='pca', learning_rate=200.0, n_jobs=-1)
    tsne_results = tsne.fit_transform(latents_list)
    title = f'Latent Invariant Representation - Real Labels'
    utils.plot.tsne_plot(tsne_results, labels_list, f'{config["plt_path"]}/{data_type}-latents-tsne.png', title)

    tsne_results = tsne.fit_transform(specific_list)
    title = f'Specific Representation - Real Labels'
    utils.plot.tsne_plot(tsne_results, labels_list, f'{config["plt_path"]}/{data_type}-specific-tsne.png', title)

    tsne_results = tsne.fit_transform(combined_list)
    title = f'Combined Representation - Real Labels'
    utils.plot.tsne_plot(tsne_results, labels_list, f'{config["plt_path"]}/{data_type}-combined-tsne.png', title)
    title = f'Combined Representation - MLP Classifier Output - Accuracy: {mlp_acc:.02f}%'
    utils.plot.tsne_plot(tsne_results, classifier_out_list, f'{config["plt_path"]}/{data_type}-combined-cls-tsne.png', title)
    title = f'Combined Representation - KNN Output - Accuracy: {knn_acc:.02f}%'
    utils.plot.tsne_plot(tsne_results, knn_output, f'{config["plt_path"]}/{data_type}-combined-knn-tsne.png', title)


def knn(encoder, specific, train_loader, device):
    encoder.eval()
    specific.eval()

    batch_acc = 0
    knn = KNN(n_neighbors=3, n_jobs=-1)
    with torch.no_grad():
        combined_list = []
        labels_list = []
        for images, labels in train_loader:
            images = images.to(device)
            latents, _, _ = encoder(images)
            specific_output = specific(images)

            combined_list.extend(torch.cat([latents, specific_output], dim=1).cpu().numpy())
            labels_list.extend(labels.cpu().tolist())

        knn.fit(combined_list, labels_list)
    
    return knn


def save_model(model, path, name):
    torch.save(model.state_dict(), f'{path}/{name}.pt')


def main(config):
    ## ------ Reproducibility ------ ##
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1)
    ## ------ Load Data ------ ###
    train_tasks, val_tasks, test_tasks = utils.data.load_tasks(config['dataset'],
                                                               config['balanced'],
                                                               val=False)
    data_shape = utils.data.get_task_data_shape(train_tasks)
    classes = utils.data.get_tasks_classes(train_tasks)
    tasks_labels = utils.data.get_tasks_labels(train_tasks)
    n_tasks = len(train_tasks)
    n_classes = len(classes)

    ### ------ Setting device ------ ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ### ------ Loading IRCL Models ------ ###
    assert config['specific_arch'] in ['MLP', 'Conv']
    assert config['encoder_arch'] in ['IRCL', 'Conv']
    assert config['decoder_arch'] in ['IRCL', 'Conv']
    if config['specific_arch'] == 'MLP':
        specific = models.mlp.Specific(data_shape, 20).to(device)
    elif config['specific_arch'] == 'Conv':
        specific = models.conv.Specific(data_shape, 20).to(device)

    if config['encoder_arch'] == 'IRCL':
        encoder = models.ircl.Encoder(data_shape, 300, config['latent_size']).to(device)
    elif config['encoder_arch'] == 'Conv':
        encoder = models.conv.Encoder(data_shape, config['latent_size']).to(device)

    if config['decoder_arch'] == 'IRCL':
        decoder = models.ircl.Decoder(data_shape, 300, config['latent_size'], n_classes).to(device)
    elif config['decoder_arch'] == 'Conv':
        decoder = models.conv.Decoder(data_shape, config['latent_size'], n_classes).to(device)

    classifier = models.mlp.Classifier(config['latent_size'], 20, 40, n_classes, softmax=config['softmax']).to(device)

    acc_of_task_t_at_time_t = []

    ### ------ Train the sequence of tasks ------ ###
    for task in range(n_tasks):
        print(f'Training Task {task}')
        train_set = copy.copy(train_tasks[task])

        if task > 0:
            gen_images, gen_labels = gen_pseudo_samples(config['n_replay'],
                                                        tasks_labels[:task],
                                                        decoder,
                                                        n_classes,
                                                        config['latent_size'],
                                                        device)

            train_set = utils.data.update_train_set(train_set,
                                                    gen_images,
                                                    gen_labels)

        train_loader = utils.data.get_dataloader(train_set,
                                                 config['batch_size'])

        val_set = val_tasks[:task + 1] if val_tasks is not None else test_tasks[:task + 1]
        val_loader = utils.data.get_dataloader(val_set, config['batch_size'])

        test_set = test_tasks[task]
        test_loader = utils.data.get_dataloader(test_set, batch_size=128)

        train_task(config, encoder, decoder, specific, classifier, train_loader, val_loader, tasks_labels[:task+1], task, device)
        acc = test(encoder, specific, classifier, test_loader, device)
        mlflow.log_metric(f'training_acc_task', acc, step=task)
        acc_of_task_t_at_time_t.append(acc)

    ### ------ Testing tasks ask after training all of them ------ ###

    ACCs = []
    BWTs = []
    for task in range(n_tasks):
        test_set = test_tasks[task]
        test_loader = utils.data.get_dataloader(test_set, batch_size=1000)
        task_acc = test(encoder, specific, classifier, test_loader, device)
        bwt = task_acc - acc_of_task_t_at_time_t[task]
        mlflow.log_metrics({f'acc_task': task_acc,
                            f'bwt_task': bwt},
                           step=task)
        ACCs.append(task_acc)
        BWTs.append(bwt)

    utils.plot.plot_losses(ACCs,
                           xlabel='Task',
                           ylabel='Accuracy',
                           title='Test Accuracy',
                           fname=f'{config["plt_path"]}/tasks-acc.png')

    avg_acc = sum(ACCs)/n_tasks
    avg_bwt = sum(BWTs)/(n_tasks-1)
    with open(config['exp_path']+'/output.log', 'w+') as f:
        for task_id, acc in enumerate(ACCs):
            print(f'Task {task_id} - Accuracy: {(acc)*100:.02f}%', file=f)

        print(f'Average accuracy: {avg_acc*100:.02f}%', file=f)
        print(f'Average backward transfer: {avg_bwt*100:.02f}%', file=f)
    print(f'Average accuracy: {avg_acc*100:.02f}%')
    print(f'Average backward transfer: {avg_bwt*100:.02f}%')


    mlflow.log_metrics({'Average accuracy': avg_acc,
                        'Average BWT': avg_bwt})

    train_loader = utils.data.get_dataloader(train_tasks, config['batch_size'])
    test_loader = utils.data.get_dataloader(test_tasks, batch_size=1000)
    knn_ = knn(encoder, specific, train_loader, device)
    sne(config['exp_path'], encoder, specific, classifier, knn_, test_loader, device, 'test')
    sne(config['exp_path'], encoder, specific, classifier, knn_, train_loader, device, 'train')


def load_config(file):
    config = None
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config


def get_path(config, idx=-1):
    folder = f'{config["specific_arch"]}S+{config["encoder_arch"]}E+{config["decoder_arch"]}D'
    subfolder = mlflow.active_run().info.run_id if config['exp_name'] == '' else config['exp_name']
    subfolder = subfolder if idx == -1 else f'{subfolder}-{idx}'
    exp_path = '/'.join([config['root'],
                         config['dataset'],
                         str(config['epochs']) + ' epochs',
                         folder,
                         subfolder])

    plt_path = '/'.join([exp_path, config['plt_path']])

    aux_path = ''
    for folder in plt_path.split('/'):
        aux_path = aux_path + folder + '/'
        if not Path(aux_path).is_dir():
            os.mkdir(aux_path)

    return exp_path


def log_config(config):
    for k, v in config.items():
        if k in ['root', 'runs', 'exp_name', 'plt_path']:
            continue
        mlflow.log_param(k, v)

if __name__ == '__main__':
    config = load_config('./config.yaml')
    assert config['runs'] > 0
    for idx in range(config['runs']):
        with mlflow.start_run(experiment_id=3):
            idx = idx if config['runs'] > 1 else -1
            config['exp_path'] = get_path(config, idx)
            os.system('cp ./config.yaml ' + '"./' + f'{config["exp_path"]}' + '"')
            config['plt_path'] = os.path.join(config['exp_path'], config['plt_path'])

            print(f'MLflow Run ID: {mlflow.active_run().info.run_id}')
            log_config(config)
            main(config)
            mlflow.log_artifacts(config['exp_path'])
