from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score
import os
import yaml
import copy
import torch
import models
import utils.data
import utils.plot
import numpy as np
import matplotlib.pyplot as plt

def gen_pseudo_samples(n_gen, tasks_labels, decoder, n_classes, latent_dim, device):
    labels = [label for labels in tasks_labels for label in labels]
    for label in labels:
        y = torch.ones(size=[n_gen], dtype=torch.int64) * label
        if label == 0:
            new_labels = y
        else:
            new_labels = torch.cat((new_labels, y))

    new_labels = new_labels[torch.randperm(new_labels.size(0))] # Shuffling the Tensor
    new_labels_onehot = utils.data.onehot_encoder(new_labels, n_classes)

    z = torch.Tensor(np.random.normal(0, 1, (n_gen * len(labels), latent_dim)))
    decoder.eval()

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


def test(encoder, specific, classifier, data_loader, device):
    encoder.eval()
    classifier.eval()
    specific.eval()

    batch_acc = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            latents, _, _ = encoder(images)

            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())

            output_list = torch.argmax(classifier_output, dim=1).tolist()
            batch_acc += accuracy_score(output_list, labels.numpy())

        batch_acc /= len(data_loader)

    return batch_acc 


def gen_recon_images(encoder, decoder, data_loader, device):
    encoder.eval()
    decoder.eval()

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
                                      lr=float(config['learning_rate']))
    optimizer_specific = torch.optim.Adam(specific.parameters(),
                                          lr=float(config['learning_rate'])/50)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(),
                                            lr=float(config['learning_rate'])/50)

    ### ------ Losses ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    classification_loss = torch.nn.CrossEntropyLoss()
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    train_bar = tqdm(range(config['epochs']))

    for epoch in train_bar:
        encoder.train()
        decoder.train()
        specific.train()
        classifier.train()
        batch_loss = 0
        batch_acc = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            ### ------ Training autoencoder ------ ###
            encoder.zero_grad()
            decoder.zero_grad()

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
            encoder.zero_grad()
            decoder.zero_grad()
            specific.zero_grad()
            classifier.zero_grad()

            #latents, _, _ = encoder(images)

            specific_output = specific(images)
            classifier_output = classifier(specific_output, latents.detach())
            classifier_loss = classification_loss(classifier_output, labels)
            classifier_loss.backward()
            optimizer_classifier.step()
            optimizer_specific.step()

            output_list = torch.argmax(classifier_output, dim=1).tolist()
            batch_acc += accuracy_score(output_list, labels.detach().cpu().numpy())
            batch_loss += cvae_loss + classifier_loss
    
        val_acc = test(encoder, specific, classifier, val_loader, device)
        
        train_bar.set_description(f'Epoch: {(epoch + 1)}/{config["epochs"]} - Loss: {(batch_loss/len(train_loader)):.03f} - Accuracy: {(batch_acc/len(train_loader))*100:.03f}% - Val Accuracy: {(val_acc)*100:.03f}%')

    real_images, recon_images = gen_recon_images(encoder, decoder, val_loader, device)
    gen_images, _ = gen_pseudo_samples(128, tasks_labels, decoder, 10, 32, device)

    utils.plot.visualize(real_images, recon_images, gen_images, task_id, config['exp_path'])


def save_model(model, path, name):
    torch.save(model.state_dict(), f'{path}/{name}.pt')


def main(config):
    torch.manual_seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    ## ------ Load Data ------ ###
    train_tasks, val_tasks, test_tasks = utils.data.load_tasks(config['dataset'],
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
    encoder = models.ircl.Encoder(data_shape, 300, config['latent_size']).to(device)
    decoder = models.ircl.Decoder(data_shape, 300, config['latent_size'], n_classes).to(device)

    ### ------ Loading convolutional Autoencoder ------ ###
    #encoder = models.conv.Encoder(data_shape, 300, config['latent_size']).to(device)
    #decoder = models.conv.Decoder(data_shape, 300, config['latent_size'], n_classes).to(device)

    ### ------ Loading Specific module and Classifier ------ ###
    specific = models.conv.Specific(data_shape, 20).to(device)
    classifier = models.conv.Classifier(config['latent_size'], 20, 40, n_classes).to(device)

    acc_of_task_t_at_time_t = []

    ### ------ Train the sequence of tasks ------ ###
    for task in range(n_tasks):
        print(f'Training Task {task}')
        train_set = train_tasks[task]

        if task > 0:
            gen_images, gen_labels = gen_pseudo_samples(config['n_replay'],
                                                        tasks_labels[:task],
                                                        decoder,
                                                        n_classes,
                                                        32,
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
        acc_of_task_t_at_time_t.append(acc)

    save_model(encoder, config['exp_path'], 'encoder')
    save_model(decoder, config['exp_path'], 'decoder')
    save_model(classifier, config['exp_path'], 'classifier')

    ### ------ Testing tasks ask after training all of them ------ ###

    ACCs = []
    BWTs = []
    for task in range(n_tasks):
        test_set = test_tasks[task]
        test_loader = utils.data.get_dataloader(test_set, batch_size=1000)
        task_acc = test(encoder, specific, classifier, test_loader, device)
        ACCs.append(task_acc)
        BWTs.append(task_acc - acc_of_task_t_at_time_t[task])


    with open(config['exp_path']+'/output.log', 'w+') as f:
        for task_id, acc in enumerate(ACCs):
            print(f'Task {task_id} - Accuracy: {(acc)*100:.02f}%', file=f)

        print(f'Average accuracy: {(sum(ACCs)/n_tasks)*100:.02f}%', file=f)
        print(f'Average backward transfer: {(sum(BWTs)/(n_tasks-1))*100:.02f}%', file=f)
    print(f'Average accuracy: {(sum(ACCs)/n_tasks)*100:.02f}%')
    print(f'Average backward transfer: {(sum(BWTs)/(n_tasks-1))*100:.02f}%')

def load_config(file):
    config = None
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config

if __name__ == '__main__':
    for idx in range(1):
        config = load_config('config.yaml')
        config['exp_path'] = config['exp_path'] + '_' + str(idx)
        if not Path(config['exp_path']).is_dir():
            os.mkdir(config['exp_path'])
        main(config)
