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

    with torch.no_grad():
        batch_acc = 0
        all_outputs = []
        all_labels = []
        for images, labels in data_loader:
            images = images.to(device)
            latents, _, _ = encoder(images)
            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())

            classifier_output = torch.argmax(classifier_output, dim=1).tolist()
            batch_acc += accuracy_score(classifier_output, labels.tolist())

            if fname:
                all_outputs.extend(classifier_output)
                all_labels.extend(labels.tolist())

        batch_acc /= len(data_loader)

    if fname is not None:
        cm = confusion_matrix(all_labels, all_outputs, labels=list(set(all_labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=list(set(all_labels)))

        disp.plot(colorbar=False)
        disp.ax_.set_title(f'Accuracy: {(batch_acc*100):.02f}%')
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

    ### ------ Loss Functions ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    classification_loss = torch.nn.CrossEntropyLoss()
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    mlflow.log_params({'loss_fn_pixelwise': 'MSE',
                       'loss_fn_classification': 'CrossEntropyLoss'})
    ### ------ Training ------ ###

    task_plt_path = f'{config["plt_path"]}/Task_{task_id}'
    if not Path(task_plt_path).is_dir():
        os.mkdir(task_plt_path)

    utils.plot.visualize_train_data(train_loader, [label for task in tasks_labels for label in task], f'{task_plt_path}/train_imgs.png')

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

            kl_loss = kl_divergence_loss(var, mu)/len(images)
            rec_loss = pixelwise_loss(recon_images, images)/len(images)
            cvae_loss = rec_loss + kl_loss

            cvae_loss.backward()
            optimizer_cvae.step()

            ### ------ Training specific module and classifier ------ ###
            encoder.eval(); decoder.eval()
            encoder.zero_grad(); decoder.zero_grad()
            specific.zero_grad(); classifier.zero_grad()

            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())
            classifier_loss = classification_loss(classifier_output, labels)/len(images)
            classifier_loss.backward()
            optimizer_classifier.step()
            optimizer_specific.step()

            output_list = torch.argmax(classifier_output, dim=1).tolist()
            epoch_acc += accuracy_score(output_list, labels.detach().cpu().numpy())

            epoch_classifier_loss += classifier_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()

        epoch_loss = epoch_classifier_loss + epoch_rec_loss + epoch_kl_loss
        mlflow.log_metrics({f'loss_rec_task_{task_id}': epoch_rec_loss/len(train_loader),
                            f'loss_kl_task_{task_id}': epoch_kl_loss/len(train_loader),
                            f'loss_cvae_task_{task_id}': (epoch_rec_loss + epoch_kl_loss)/len(train_loader),
                            f'loss_classifier_task_{task_id}': epoch_classifier_loss/len(train_loader),
                            f'loss_total_task_{task_id}': epoch_loss/len(train_loader)},
                          step=epoch)
        rec_loss_epochs.append(epoch_rec_loss/len(train_loader))
        kl_loss_epochs.append(epoch_kl_loss/len(train_loader))
        cvae_loss_epochs.append((epoch_rec_loss + epoch_kl_loss)/len(train_loader))
        classifier_loss_epochs.append(epoch_classifier_loss/len(train_loader))
        total_loss_epochs.append((epoch_loss/len(train_loader)))
        val_acc = test(encoder, specific, classifier, val_loader, device, f'{task_plt_path}/epoch_{epoch}.png')
        
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
    encoder.eval(); specific.eval()

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
    train_tasks = utils.data.load_tasks(config['dataset'], config['balanced'], True)
    test_tasks = utils.data.load_tasks(config['dataset'], config['balanced'], False)

    img_shape = utils.data.get_img_shape(train_tasks)
    classes = utils.data.get_classes(train_tasks)
    labels = utils.data.get_labels(train_tasks)
    n_tasks = len(train_tasks)
    n_classes = len(classes)

    ### ------ Setting device ------ ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ### ------ Loading Models ------ ###
    specific = models.utils.load_specific_module(img_shape, config)
    encoder = models.utils.load_encoder(img_shape, config)
    decoder = models.utils.load_decoder(img_shape, n_classes, config)
    classifier = models.utils.load_classifier(n_classes, config)
    specific.to(device); encoder.to(device); decoder.to(device); classifier.to(device)

    ### ------ Train the sequence of tasks ------ ###
    acc_of_task_t_at_time_t = []
    train_sets_tasks = []
    for task in range(n_tasks):
        print(f'Training Task {task}')
        train_set = copy.copy(train_tasks[task])

        if task > 0:
            gen_data = gen_pseudo_samples(config['n_replay'],
                                          labels[:task],
                                          decoder,
                                          n_classes,
                                          config['latent_size'],
                                          device)

            gen_images, gen_labels = gen_data
            train_set = utils.data.update_train_set(train_set,
                                                    gen_images,
                                                    gen_labels)
        train_sets_tasks.append(train_set)

        train_loader = utils.data.get_dataloader(train_set,
                                                 config['batch_size'])

        val_set = copy.copy(test_tasks[:task + 1])
        val_loader = utils.data.get_dataloader(val_set, config['batch_size'])

        train_task(config, encoder, decoder, specific, classifier, train_loader, val_loader, labels[:task+1], task, device)

        test_set = copy.copy(test_tasks[task])
        test_loader = utils.data.get_dataloader(test_set, batch_size=128)
        acc = test(encoder, specific, classifier, test_loader, device)
        acc_of_task_t_at_time_t.append(acc)
        mlflow.log_metric(f'training_acc_task', acc, step=task)

    ### ------ Testing tasks ask after training all of them ------ ###

    ACCs_test_set = []
    BWTs_test_set = []

    test_set = test_tasks
    test_loader = utils.data.get_dataloader(test_set, batch_size=1000)
    acc_train = test(encoder, specific, classifier, test_loader, device, fname=f'{config["plt_path"]}/test_test_set.png')

    for task in range(n_tasks):
        test_set = copy.copy(test_tasks[task])
        test_loader = utils.data.get_dataloader(test_set, batch_size=1000)
        acc_test = test(encoder, specific, classifier, test_loader, device)
        bwt_test = acc_test - acc_of_task_t_at_time_t[task]

        train_loader = utils.data.get_dataloader(train_sets_tasks[task], batch_size=1000)
        acc_train = test(encoder, specific, classifier, train_loader, device, fname=f'{config["plt_path"]}/test_train_set_task_{task}.png')

        mlflow.log_metrics({f'Task Accuracy Test Set': acc_test,
                            f'Task BWT Test': bwt_test},
                           step=task)

        ACCs_test_set.append(acc_test)
        BWTs_test_set.append(bwt_test)

    utils.plot.plot_losses(ACCs_test_set,
                           xlabel='Task',
                           ylabel='Accuracy',
                           title='Test Accuracy',
                           fname=f'{config["plt_path"]}/tasks-acc.png')

    avg_acc_test_set = sum(ACCs_test_set)/n_tasks
    avg_bwt_test_set = sum(BWTs_test_set)/(n_tasks-1)

    mlflow.log_metrics({'Average Test Set Accuracy': avg_acc_test_set,
                        'Average BWT': avg_bwt_test_set})

    with open(config['exp_path']+'/output.log', 'w+') as f:
        for file in [None, f]:
            print('', file=None)
            for task_id, acc_test in enumerate(ACCs_test_set):
                print(f'Task {task_id} - Test Set Accuracy: {(acc_test)*100:.02f}%', file=file)

            print('', file=file)
            print(f'Average Test Set Accuracy: {avg_acc_test_set*100:.02f}%', file=file)
            print(f'Average Test Set Backward Transfer: {avg_bwt_test_set*100:.02f}%\n', file=file)

    test_loader = utils.data.get_dataloader(test_tasks, batch_size=1000)
    train_loader = utils.data.get_dataloader(train_tasks, config['batch_size'])
    knn_ = knn(encoder, specific, train_loader, device)
    sne(config['exp_path'], encoder, specific, classifier, knn_, test_loader, device, 'test')
    sne(config['exp_path'], encoder, specific, classifier, knn_, train_loader, device, 'train')


def load_config(file):
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    assert config['runs'] > 0
    assert config['arch_specific'] in ['MLP', 'Conv']
    assert config['arch_encoder'] in ['IRCL', 'Conv']
    assert config['arch_decoder'] in ['IRCL', 'Conv']

    return config


def get_exp_path(config, idx=-1):
    folder = f'{config["arch_specific"]}S+{config["arch_encoder"]}E+{config["arch_decoder"]}D'
    subfolder = mlflow.active_run().info.run_id if config['exp_name'] == '' else config['exp_name']
    subfolder = subfolder if idx == -1 else f'{subfolder}-{idx}'

    exp_path = '/'.join([config['root'],
                         config['dataset'],
                         f'{config["epochs"]} epochs',
                         folder,
                         subfolder])

    plt_path = '/'.join([exp_path, config['plt_path']])

    aux_path = ''
    for sub_dir in plt_path.split('/'):
        aux_path += sub_dir + '/'

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

    for idx in range(config['runs']):
        with mlflow.start_run(experiment_id=4):
            idx = idx if config['runs'] > 1 else -1
            config['exp_path'] = get_exp_path(config, idx)
            os.system('cp ./config.yaml ' + '"./' + f'{config["exp_path"]}' + '"')
            config['plt_path'] = os.path.join(config['exp_path'], config['plt_path'])

            print(f'MLflow Run ID: {mlflow.active_run().info.run_id}')
            log_config(config)
            main(config)
            mlflow.log_artifacts(config['exp_path'])
