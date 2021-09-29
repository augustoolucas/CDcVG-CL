from tqdm import tqdm
import torch
import yaml
import data_utils
import ircl_models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score

def onehot_encoder(labels, n_classes):
    labels_onehot = torch.zeros(labels.shape[0], n_classes)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot

def gen_pseudo_samples(n_gen, tasks_labels, decoder, n_classes, latent_dim, device):
    if not tasks_labels:
        # This is expected to happen only on the first task.
        empty_tensor = torch.empty(0).to('cpu')
        return empty_tensor, empty_tensor

    labels = [label for labels in tasks_labels for label in labels]
    z = torch.Tensor(np.random.normal(0, 1, (n_gen * len(labels), latent_dim)))
    for label in labels:
        y = torch.ones(size=[n_gen], dtype=torch.int64) * label
        if label == 0:
            new_labels = y
        else:
            new_labels = torch.cat((new_labels, y))

    new_labels = new_labels[torch.randperm(new_labels.size(0))] # Shuffling the Tensor
    new_labels_onehot = onehot_encoder(new_labels, n_classes)
    decoder.eval()
    with torch.no_grad():
        input = torch.cat([z, torch.Tensor(new_labels_onehot)], dim=1)
        new_images = decoder(input.to(device))

    return new_images.to('cpu'), new_labels.to('cpu')

def train(config, encoder, decoder, classifier, train_loader, val_loader, task_labels, device):
    ### ------ Optimizers ------ ###
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=float(config['learning_rate']))
    optimizer_C = torch.optim.Adam(classifier.parameters(),
                                   lr=float(config['learning_rate'])/50)

    ### ------ Losses ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    classification_loss = torch.nn.CrossEntropyLoss()
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    encoder.train()
    decoder.train()
    classifier.train()
    train_bar = tqdm(range(config['epochs']))

    for epoch in train_bar:
        total_loss = 0
        acc = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            ### ------ Training autoencoder ------ ###
            encoder.zero_grad()
            decoder.zero_grad()

            labels_onehot = onehot_encoder(labels, 10).to(device)
            images, labels = images.to(device), labels.to(device)
            latents, mu, var = encoder(images)
            recon_images = decoder(torch.cat([latents, labels_onehot], dim=1))

            kl_loss = kl_divergence_loss(var, mu)/config['batch_size']
            rec_loss = pixelwise_loss(recon_images, images)/config['batch_size']
            cvae_loss = rec_loss + kl_loss

            cvae_loss.backward()
            optimizer_cvae.step()

            ### ------ Training classifier ------ ###
            classifier.zero_grad()

            classifier_output = classifier(images.view(images.size(0), -1),
                                           latents.detach())
            classifier_loss = classification_loss(classifier_output, labels)

            output_list = torch.argmax(classifier_output, dim=1).tolist()
            acc += accuracy_score(output_list, labels.detach().cpu().numpy())
            classifier_loss.backward()
            optimizer_C.step()

            total_loss += cvae_loss + classifier_loss

            train_bar.set_description(
                f'Epoch: {(epoch + 1)}/{config["epochs"]} - Loss: {(total_loss/batch_idx):.03f} - Accuracy: {(acc/batch_idx)*100:.03f}%'
            )
    
        total_loss /= len(train_loader)
    # evaluate using validation set
    # visualize 



def main(config):
    ### ------ Load Data ------ ###
    train_tasks, val_tasks, test_tasks = data_utils.load_tasks('MNIST',
                                                               val=True)
    data_shape = data_utils.get_task_data_shape(train_tasks)
    n_classes = data_utils.get_task_n_classes(train_tasks)
    tasks_labels = data_utils.get_tasks_labels(n_classes)
    n_tasks = len(train_tasks)

    ### ------ Setting device ------ ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ### ------ Loading IRCL Models ------ ###
    encoder, decoder, classifier = ircl_models.load_all(data_shape,
                                                        n_classes,
                                                        device)

    ### ------ Train the sequence of CL tasks ------ ###
    for task in range(n_tasks):
        print(f'Training Task {task}')
        train_set = train_tasks[task]
        val_set = val_tasks[task]

        gen_images, gen_labels = gen_pseudo_samples(config['n_replay'],
                                                    tasks_labels[:task],
                                                    decoder,
                                                    n_classes,
                                                    32,
                                                    device)

        train_set = data_utils.update_train_set(train_set, gen_images, gen_labels)

        train_loader = data_utils.get_dataloader(train_set, config['batch_size'])
        val_loader = data_utils.get_dataloader(val_set, config['batch_size'])

        acc = train(config, encoder, decoder, classifier, train_loader, val_loader, tasks_labels[:task], device)


def load_config(file):
    config = None
    with open(file, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.SafeLoader)

    return config

if __name__ == '__main__':
    config = load_config('config.yaml')
    main(config)
