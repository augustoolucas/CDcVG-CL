import os
import yaml
import copy
import torch
import mlflow
import models
import random
import optuna
import joblib
import pickle
import utils.data
import utils.plot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.cuda import amp
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.manifold import TSNE
from models.utils import DEVICE
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

def gen_pseudo_samples(n_samples, labels, decoder, n_classes, latent_dim, use_amp=False):
    decoder.eval()

    labels = [label for labels in labels for label in labels]
    for label in labels:
        y = torch.ones(size=[n_samples], dtype=torch.int64) * label
        new_labels = y if label == labels[0] else torch.cat((new_labels, y))

    new_labels = new_labels[torch.randperm(new_labels.size(0))] # Shuffling the Tensor
    new_labels_onehot = utils.data.onehot_encoder(new_labels, n_classes)

    z = torch.Tensor(np.random.normal(0, 1, (n_samples * len(labels), latent_dim)))

    with torch.no_grad():
        input = torch.cat([z, new_labels_onehot], dim=1)
        input = torch.split(input, 128)
        new_images = torch.Tensor([])

        for chunk in input:
            with amp.autocast(enabled=use_amp):
                gen_images = decoder(chunk.to(DEVICE))
            gen_images = gen_images.to('cpu')
            new_images = torch.cat([new_images, gen_images])
        torch.cuda.empty_cache()

    return new_images, new_labels


def gen_recon_images(encoder, decoder, data_loader, use_amp=False):
    encoder.eval(); decoder.eval()

    ### Generate reconstructed images ###
    with torch.no_grad():
        images, labels = next(iter(data_loader))
        labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
        images = images.to(DEVICE)
        with amp.autocast(enabled=use_amp):
            latents, _, _ = encoder(images)
            recon_images = decoder(torch.cat([latents, labels_onehot], dim=1)).to('cpu')

        images = images.to('cpu')

    return images, recon_images


def get_features(encoder, specific, imgs, use_amp=False):
    encoder.eval(); specific.eval()

    all_z = torch.empty(size=(0, 32))
    all_specific = torch.empty(size=(0, 20))
    all_combined = torch.empty(size=(0, 52))
    with torch.no_grad():
        for img in imgs:
            with amp.autocast(enabled=use_amp):
                latent, _, _ = encoder(img.unsqueeze(0).to(DEVICE))
            all_z = torch.cat((all_z, latent.to('cpu')))
            all_specific = torch.cat((all_specific, specific(img.unsqueeze(0).to(DEVICE)).to('cpu')))
            all_combined = torch.cat((all_combined,
                                      torch.cat((all_specific[-1], all_z[-1])).unsqueeze(0)))

    return all_z, all_specific, all_combined


def train_task(config, encoder, decoder, specific, classifier, train_loader, tasks_labels, task_id, discriminator=None):
    scaler = amp.GradScaler(enabled=config['use_amp'])
    ### ------ Optimizers ------ ###
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=float(config['lr_autoencoder']))
    optimizer_specific = torch.optim.Adam(specific.parameters(),
                                          lr=float(config['lr_specific']))
    optimizer_classifier = torch.optim.Adam(classifier.parameters(),
                                            lr=float(config['lr_classifier']))

    if discriminator is not None:
        optimizer_discriminator = torch.optim.Adam(
                                        discriminator.parameters(),
                                        lr=float(config['lr_discriminator'])
                                        )
        discriminator_loss = torch.nn.BCEWithLogitsLoss()
        discriminator.train()
        real_label, recon_label = 1, 0

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
        classes_count = defaultdict(int)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            for cls in set(labels.tolist()):
                classes_count[cls] += len(labels[labels == cls])
            ### ------ Training autoencoder ------ ###
            encoder.train(); decoder.train();
            encoder.zero_grad(); decoder.zero_grad()

            labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with amp.autocast(enabled=config['use_amp']):
                latents, mu, var = encoder(images)

                if discriminator is not None:
                    discriminator.zero_grad()
                    disc_output = discriminator(images).view(-1)
                    disc_labels = torch.full((len(images),),
                                             real_label,
                                             dtype=torch.float,
                                             device=DEVICE)
                    lossD_real = discriminator_loss(disc_output, disc_labels)
                    with amp.autocast(enabled=False):
                        scaler.scale(lossD_real).backward()

                recon_images = decoder(torch.cat([latents, labels_onehot], dim=1))

                if discriminator is not None:
                    disc_output = discriminator(recon_images.detach()).view(-1)
                    disc_labels.fill_(recon_label)
                    lossD_recon = discriminator_loss(disc_output, disc_labels)
                    with amp.autocast(enabled=False):
                        scaler.scale(lossD_recon).backward()
                        scaler.step(optimizer_discriminator)

                if discriminator is not None:
                    decoder.zero_grad()
                    disc_output = discriminator(recon_images).view(-1)
                    disc_labels.fill_(real_label)
                    lossG = discriminator_loss(disc_output, disc_labels)

                kl_loss = kl_divergence_loss(var, mu)/len(images)
                rec_loss = pixelwise_loss(recon_images, images)/len(images)
                cvae_loss = rec_loss + kl_loss

            if discriminator is not None:
                cvae_loss += lossG

            scaler.scale(cvae_loss).backward()
            scaler.step(optimizer_cvae)

            ### ------ Training specific module and classifier ------ ###
            specific.train(); classifier.train()
            specific.zero_grad(); classifier.zero_grad()

            with amp.autocast(enabled=config['use_amp']):
                specific_output = specific(images)
                classifier_output = classifier(specific_output, latents.detach())
                classifier_loss = classification_loss(classifier_output, labels)/len(images)
            scaler.scale(classifier_loss).backward()
            scaler.step(optimizer_classifier)
            scaler.step(optimizer_specific)
            scaler.update()

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
        #val_acc = test(encoder, specific, classifier, val_loader, f'{task_plt_path}/epoch_{epoch}.png')

        train_bar.set_description(f'Epoch: {(epoch + 1)}/{config["epochs"]} - '
                                  f'Loss: {(epoch_loss/len(train_loader)):.03f} - '
                                  f'Accuracy: {(epoch_acc/len(train_loader))*100:.03f}% - ')
                                  #f'Val Accuracy: {(val_acc)*100:.03f}%')

    fig = plt.figure()
    plt.bar(list(classes_count.keys()), list(classes_count.values()))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, key in enumerate(classes_count):
        ax.annotate(classes_count[key],
                    xy=(idx-len(classes_count)*0.035, classes_count[key]+(max(classes_count.values()) - min(classes_count.values()))*0.025),
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


def sne(path, encoder, specific, classifier, knn, data_loader, data_type):
    encoder.eval(); classifier.eval(); specific.eval()

    with torch.no_grad():
        specific_list = []
        latents_list = []
        labels_list = []
        combined_list = []
        classifier_out_list = []
        for images, labels in data_loader:
            images = images.to(DEVICE)
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


def train_mlp(config, encoder, specific, classifier, data_loader):
    encoder.eval(); specific.eval(); classifier.train()

    optimizer_classifier = torch.optim.Adam(classifier.parameters(),
                                            lr=0.001)

    ### ------ Loss Function ------ ###
    classification_loss = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(20)):
        epoch_loss = 0
        for images, labels in data_loader:
            classifier.zero_grad()
            with torch.no_grad():
                images = images.to(DEVICE)
                latents, _, _ = encoder(images)
                specific_output = specific(images)

            labels = labels.to(DEVICE)
            classifier_output = classifier(specific_output, latents.detach())
            classifier_loss = classification_loss(classifier_output, labels)/len(images)
            classifier_loss.backward()
            optimizer_classifier.step()
            epoch_loss += classifier_loss.item()

        mlflow.log_metric('MLP Classifier Loss', epoch_loss, step=epoch)

    return classifier


def knn(encoder, specific, train_loader, use_amp=False):
    encoder.eval(); specific.eval()

    knn = KNN(n_neighbors=3, n_jobs=-1)
    with torch.no_grad():
        combined_list = []
        labels_list = []
        for images, labels in train_loader:
            images = images.to(DEVICE)
            with amp.autocast(enabled=use_amp):
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
    train_tasks = utils.data.load_tasks(config['dataset'], config['balanced'], train=True)
    test_tasks = utils.data.load_tasks(config['dataset'], config['balanced'], train=False)

    img_shape = utils.data.get_img_shape(train_tasks)
    classes = utils.data.get_classes(train_tasks)
    labels = utils.data.get_labels(train_tasks)
    n_tasks = len(train_tasks)
    n_classes = len(classes)

    ### ------ Loading Models ------ ###
    specific = models.utils.load_specific_module(img_shape, config)
    encoder = models.utils.load_encoder(img_shape, config)
    decoder = models.utils.load_decoder(img_shape, n_classes, config)
    classifier = models.utils.load_classifier(n_classes, config)

    discriminator = None
    if config['use_discriminator']:
        discriminator = models.utils.load_discriminator(img_shape, config)

    ### ------ Train the sequence of tasks ------ ###

    acc_of_task_t_at_time_t = []
    train_sets_tasks = []
    for task in range(n_tasks):
        print(f'Training Task {task}')

        task_plt_path = f'{config["plt_path"]}/Task_{task}'
        if not Path(task_plt_path).is_dir():
            os.mkdir(task_plt_path)

        train_set = copy.copy(train_tasks[task])

        if task > 0:
            gen_images, gen_labels = gen_pseudo_samples(
                                                n_samples=config['n_replay'],
                                                labels=labels[:task],
                                                decoder=decoder,
                                                n_classes=n_classes,
                                                latent_dim=config['latent_size']
                                                )

            train_set = utils.data.update_train_set(train_set,
                                                    gen_images,
                                                    gen_labels)
        train_sets_tasks.append(train_set)

        train_loader = utils.data.get_dataloader(train_set,
                                                 config['batch_size'])

        val_set = copy.copy(test_tasks[:task + 1])
        val_loader = utils.data.get_dataloader(val_set, config['batch_size'])

        if config['decoupled_cvae_training']:
            models.utils.train_vaegan(config=config,
                                      encoder=encoder,
                                      decoder=decoder,
                                      discriminator=discriminator,
                                      data_loader=train_loader,
                                      task_id=task)

            models.utils.train_classifier(config=config,
                                          encoder=encoder,
                                          specific=specific,
                                          classifier=classifier,
                                          data_loader=train_loader,
                                          task_id=task)

        else:
            train_task(config=config,
                       encoder=encoder,
                       decoder=decoder,
                       specific=specific,
                       classifier=classifier,
                       discriminator=discriminator,
                       train_loader=train_loader,
                       tasks_labels=labels[:task+1],
                       task_id=task)

        models.utils.test(encoder=encoder,
             specific=specific,
             classifier=classifier,
             data_loader=val_loader,
             use_amp=config['use_amp'],
             fname=f'{task_plt_path}/test_set.png')

        models.utils.test(encoder=encoder,
             specific=specific,
             classifier=classifier,
             data_loader=train_loader,
             use_amp=config['use_amp'],
             fname=f'{task_plt_path}/train_set.png')

        test_set = copy.copy(test_tasks[task])
        test_loader = utils.data.get_dataloader(test_set, batch_size=128)
        acc = models.utils.test(encoder=encoder,
                   specific=specific,
                   classifier=classifier,
                   data_loader=test_loader,
                   use_amp=config['use_amp'])

        acc_of_task_t_at_time_t.append(acc)
        mlflow.log_metric(f'training_acc_task', acc, step=task)

        real_images, recon_images = gen_recon_images(encoder,
                                                     decoder,
                                                     val_loader)
        gen_images, _ = gen_pseudo_samples(n_samples=128,
                                           labels=labels[:task+1],
                                           decoder=decoder,
                                           n_classes=n_classes,
                                           latent_dim=config['latent_size'])
        utils.plot.visualize(real_images,
                             recon_images,
                             gen_images,
                             f'{task_plt_path}/images.png')

    ### ------ Testing tasks ask after training all of them ------ ###
    ACCs_test_set = []
    BWTs_test_set = []

    test_set = test_tasks
    test_loader = utils.data.get_dataloader(test_set, batch_size=1000)
    acc_train = models.utils.test(
                                encoder,
                                specific,
                                classifier,
                                test_loader,
                                use_amp=config['use_amp'],
                                fname=f'{config["plt_path"]}/test_test_set.png'
                                )

    for task in range(n_tasks):
        test_set = copy.copy(test_tasks[task])
        test_loader = utils.data.get_dataloader(test_set, batch_size=1000)
        acc_test = models.utils.test(encoder,
                                     specific,
                                     classifier,
                                     test_loader,
                                     use_amp=config['use_amp'])
        bwt_test = acc_test - acc_of_task_t_at_time_t[task]

        train_loader = utils.data.get_dataloader(train_sets_tasks[task], batch_size=1000)
        acc_train = models.utils.test(
                                    encoder,
                                    specific,
                                    classifier,
                                    train_loader,
                                    use_amp=config['use_amp'],
                                    fname=f'{config["plt_path"]}/test_train_set_task_{task}.png'
                                    )

        mlflow.log_metrics({f'Task Accuracy Test Set': acc_test,
                            f'Task BWT Test': bwt_test},
                           step=task)

        ACCs_test_set.append(acc_test)
        BWTs_test_set.append(bwt_test)

    ### ------ Logging ------ ###
    utils.plot.plot_losses(ACCs_test_set,
                           xlabel='Task',
                           ylabel='Accuracy',
                           title='Test Accuracy',
                           fname=f'{config["plt_path"]}/tasks-acc.png')

    avg_acc_test_set = np.average(ACCs_test_set)
    stdev_acc_test_set = np.std(ACCs_test_set)
    avg_bwt_test_set = sum(BWTs_test_set)/(n_tasks-1)

    mlflow.log_metrics({'Average Test Set Accuracy': avg_acc_test_set,
                        'Average BWT': avg_bwt_test_set})

    with open(config['exp_path']+'/output.log', 'w+') as f:
        for file in [None, f]:
            print('', file=None)
            for task_id, acc_test in enumerate(ACCs_test_set):
                print(f'Task {task_id} - Test Set Accuracy: {(acc_test)*100:.02f}%',
                      file=file)

            print('', file=file)
            print(f'Average Test Set Accuracy: {avg_acc_test_set*100:.02f}%',
                  file=file)
            print(f'Average Test Set Backward Transfer: {avg_bwt_test_set*100:.02f}%\n',
                  file=file)

    test_loader = utils.data.get_dataloader(test_tasks, batch_size=1000)
    train_loader = utils.data.get_dataloader(train_tasks, config['batch_size'])
    if config['plot_tsne']:
        knn_ = knn(encoder, specific, train_loader)
        sne(config['exp_path'],
            encoder,
            specific,
            classifier,
            knn_,
            test_loader,
            'test')
        sne(config['exp_path'],
            encoder,
            specific,
            classifier,
            knn_,
            train_loader,
            'train')

    """
    cls = train_mlp(config, encoder, specific, classifier, train_loader)
    acc = test(encoder, specific, cls, test_loader)

    with open(f'{config["exp_path"]}/output.log', 'a') as f:
        print(f'MLP Classifier Accuracy: {(acc*100):.02f}%', file=f)
    print(f'MLP Classifier Accuracy: {(acc*100):.02f}%')
    """

    if config['save_images']:
        img_path = f'{config["exp_path"]}/real_images'
        os.makedirs(img_path)
        for task in range(n_tasks):
            utils.plot.save_images(train_tasks[task].data,
                                   train_tasks[task].targets,
                                   img_path)

        img_path = f'{config["exp_path"]}/generated_images'
        os.makedirs(img_path)

    for task in range(n_tasks):
        all_z = torch.empty(size=(0, 32))
        all_specific = torch.empty(size=(0, 20))
        all_combined = torch.empty(size=(0, 52))
        all_labels = []
        z, specif, combined = get_features(
                                        encoder,
                                        specific,
                                        [img for img, _ in test_tasks[task]],
                                        config['use_amp']
                                        )

        all_z = torch.cat((all_z, z))
        all_specific = torch.cat((all_specific, specif))
        all_combined = torch.cat((all_combined, combined))
        all_labels.extend(test_tasks[task].targets.tolist())

        gen_images, gen_labels = gen_pseudo_samples(
                                            n_samples=1024,
                                            labels=[labels[task]],
                                            decoder=decoder,
                                            n_classes=n_classes,
                                            latent_dim=config['latent_size'],
                                            use_amp=config['use_amp']
                                            )

        z, specif, combined = get_features(encoder,
                                           specific,
                                           gen_images,
                                           config['use_amp'])
        all_z = torch.cat((all_z, z))
        all_specific = torch.cat((all_specific, specif))
        all_combined = torch.cat((all_combined, combined))
        all_labels.extend((10 + gen_labels).tolist())
        k = lambda x: f'{x} Real' if x < 10 else f'{x - 10} Generated'
        keys = {idx: k(idx) for idx in all_labels}
        all_labels = [keys[idx] for idx in all_labels]

        tsne = TSNE(n_components=2,
                    verbose=0,
                    perplexity=40,
                    n_iter=300,
                    init='pca',
                    learning_rate=200.0,
                    n_jobs=-1)
        tsne_results = tsne.fit_transform(all_combined.detach().numpy())
        utils.plot.tsne_plot(tsne_results,
                             all_labels,
                             f'{config["plt_path"]}/combined_task_{task}.png')

        if config['save_images']:
            utils.plot.save_images(gen_images, gen_labels, img_path)

    return avg_acc_test_set, stdev_acc_test_set

def load_config(file):
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    assert config['runs'] > 0
    assert config['arch_specific'] in ['MLP', 'Conv']
    assert config['arch_encoder'] in ['IRCL', 'Conv']
    assert config['arch_decoder'] in ['IRCL', 'Conv']

    return config


def get_exp_path(config):
    run_id = mlflow.active_run().info.run_id
    exp_path = '/'.join([config['root'], run_id])
    plt_path = '/'.join([exp_path, config['plt_path']])
    os.makedirs(plt_path)

    return exp_path


def log_config(config):
    log_config = {k: v for k,v in config.items() if k not in ['root',
                                                              'runs',
                                                              'exp_name',
                                                              'plt_path',
                                                              'exp_path']}
    mlflow.log_params(log_config)

def run(trial=None):
    with mlflow.start_run(experiment_id=18, run_name=''):
        config = load_config('./config.yaml')
        config['exp_path'] = get_exp_path(config)
        os.system(f'cp ./config.yaml ./{config["exp_path"]}/')
        config['plt_path'] = os.path.join(config['exp_path'],
                                          config['plt_path'])

        print(f'MLflow Run ID: {mlflow.active_run().info.run_id}')

        ### ------ Optuna ------ ###
        if trial is not None:
            TRIED_CFGS = f'Discriminator-Conv-{config["dataset"]}-cfgs.pkl'
            config['lr_specific'] = trial.suggest_categorical(
                                                    'lr_specific',
                                                    [1e-3, 1e-4, 1e-5]
                                                    )
            config['lr_classifier'] = trial.suggest_categorical(
                                                    'lr_classifier',
                                                    [1e-3, 1e-4, 1e-5]
                                                    )
            config['lr_autoencoder'] = trial.suggest_categorical(
                                                    'lr_autoencoder',
                                                    [1e-3, 1e-4, 1e-5]
                                                    )
            config['lr_discriminator'] = trial.suggest_categorical(
                                                    'lr_discriminator',
                                                    [1e-3, 1e-4, 1e-5]
                                                    )
            if config['decoupled_cvae_training'] is True:
                config['epochs_classifier'] = trial.suggest_categorical(
                                                'epochs_classifier',
                                                [5, 10, 15, 20, 25, 30, 40, 50]
                                                )
                config['epochs_cvae'] = trial.suggest_categorical(
                                                'epochs_cvae',
                                                [5, 10, 15, 20, 25, 50, 75,
                                                 100, 200]
                                                )
            else:
                config['epochs'] = trial.suggest_categorical(
                                                'epochs',
                                                [5, 10, 15, 20, 25, 30, 40, 50]
                                                )
            if not os.path.exists(TRIED_CFGS):
                with open(TRIED_CFGS, 'wb+') as f:
                    pickle.dump([config], f, )
            else:
                with open(TRIED_CFGS, 'rb') as f:
                    pkl = pickle.load(f)
                if config in pkl:
                    raise optuna.exceptions.TrialPruned()
                pkl.append(config)
                with open(TRIED_CFGS, 'wb+') as f:
                    pickle.dump(pkl, f)

        log_config(config)
        avg_acc, _ = main(config)
        mlflow.log_artifacts(config['exp_path'])

        if trial is not None:
            joblib.dump(study, STUDY_FILE)

    return avg_acc


if __name__ == '__main__':
    config = load_config('./config.yaml')
    STUDY_FILE = f'DCGAN-Disc-{config["dataset"]}.pkl'
    if config['optuna']:
        if not os.path.exists(STUDY_FILE):
            study = optuna.create_study(directions=['maximize'])
        else:
            study = joblib.load(STUDY_FILE)

        study.optimize(run, timeout=3600*(21))
        joblib.dump(study, STUDY_FILE)
        print("Number of finished trials: ", len(study.trials))
    else:
        run()
