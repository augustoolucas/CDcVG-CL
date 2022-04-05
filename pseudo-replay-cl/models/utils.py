import utils
import torch
import models
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda import amp
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_specific_module(img_shape, config):
    if config['arch_specific'] == 'MLP':
        specific = models.mlp.Specific(img_shape=img_shape, specific_size=20)
    elif config['arch_specific'] == 'Conv':
        specific = models.conv.Specific(img_shape=img_shape, specific_size=20)

    return specific

def load_encoder(img_shape, config):
    if config['arch_encoder'] == 'IRCL':
        encoder = models.ircl.Encoder(img_shape=img_shape,
                                      n_hidden=300,
                                      latent_dim=config['latent_size'])
    elif config['arch_encoder'] == 'Conv':
        encoder = models.conv.Encoder(img_shape=img_shape,
                                      latent_dim=config['latent_size'])

    return encoder

def load_decoder(img_shape, n_classes, config):
    if config['arch_decoder'] == 'IRCL':
        decoder = models.ircl.Decoder(img_shape,
                                      300,
                                      config['latent_size'],
                                      n_classes)
    elif config['arch_decoder'] == 'Conv':
        decoder = models.conv.Decoder(img_shape, config['latent_size'], n_classes)

    return decoder

def load_classifier(n_classes, config):
    return models.mlp.Classifier(invariant_size=config['latent_size'],
                                 specific_size=20,
                                 classification_n_hidden=40,
                                 n_classes=n_classes,
                                 softmax=config['softmax'])


def train_cvae(config, encoder, decoder, data_loader, task_id=None):
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=float(config['lr_autoencoder']))

    ### ------ Loss Functions ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    mlflow.log_params({'optimizer_cvae': 'Adam',
                       'loss_fn_pixelwise': 'MSE'})

    train_bar = tqdm(range(config['epochs_cvae']))

    for epoch in train_bar:
        encoder.train(); decoder.train();
        epoch_rec_loss, epoch_kl_loss = 0, 0
        for images, labels in data_loader:
            encoder.zero_grad(); decoder.zero_grad()

            labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
            images = images.to(DEVICE)
            latents, mu, var = encoder(images)
            recon_images = decoder(torch.cat([latents, labels_onehot], dim=1))

            kl_loss = kl_divergence_loss(var, mu)/len(images)
            rec_loss = pixelwise_loss(recon_images, images)/len(images)
            cvae_loss = rec_loss + kl_loss

            cvae_loss.backward()
            optimizer_cvae.step()

            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()

        rec_loss_str = f'Loss Reconstruction Task {task_id}' if task_id is not None else 'Loss Reconstruction'
        kl_loss_str = f'Loss KL Task {task_id}' if task_id is not None else 'Loss KL'
        cvae_loss_str = f'Loss CVAE Task {task_id}' if task_id is not None else 'Loss CVAE'
        mlflow.log_metrics({rec_loss_str: epoch_rec_loss/len(data_loader),
                            kl_loss_str: epoch_kl_loss/len(data_loader),
                            cvae_loss_str: (epoch_rec_loss + epoch_kl_loss)/len(data_loader)},
                           step=epoch)


def train_classifier(config, encoder, specific, classifier, data_loader, task_id=None):
    scaler = amp.GradScaler(enabled=config['use_amp'])
    encoder.eval(); specific.train(); classifier.train()
    optimizer_specific = torch.optim.Adam(specific.parameters(),
                                          lr=float(config['lr_specific']))
    optimizer_classifier = torch.optim.Adam(classifier.parameters(),
                                            lr=float(config['lr_classifier']))

    ### ------ Loss Function ------ ###
    classification_loss = torch.nn.CrossEntropyLoss()

    mlflow.log_params({'optimizer_specific': 'Adam',
                       'optimizer_classifier': 'Adam',
                       'loss_fn_classification': 'CrossEntropyLoss'})

    train_bar = tqdm(range(config['epochs_classifier']))

    for epoch in train_bar:
        epoch_classifier_loss, epoch_acc = 0, 0
        classifier.train()
        for batch_idx, (images, labels) in enumerate(data_loader, start=1):
            specific.zero_grad(); classifier.zero_grad()

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with amp.autocast(enabled=config['use_amp']):
                with torch.no_grad():
                    latents, _, _ = encoder(images)

                specific_embedding = specific(images)
                classifier_output = classifier(specific_embedding, latents.detach())
                classifier_loss = classification_loss(classifier_output, labels)/len(images)

            scaler.scale(classifier_loss).backward()
            scaler.step(optimizer_classifier)
            scaler.step(optimizer_specific)
            scaler.update()

            output_list = torch.argmax(classifier_output, dim=1).tolist()
            epoch_acc += accuracy_score(output_list, labels.detach().cpu().numpy())
            epoch_classifier_loss += classifier_loss.item()

        cls_loss_str = f'Loss Classifier Task {task_id}' if task_id is not None else 'Loss Classifier' 
        mlflow.log_metrics({cls_loss_str: epoch_classifier_loss/len(data_loader)},
                           step=epoch)

        for _ in range(0):
            classifier.eval()
            for batch_idx, (images, labels) in enumerate(data_loader, start=1):
                specific.zero_grad(); classifier.zero_grad()

                images, labels = images.to(DEVICE), labels.to(DEVICE)

                with torch.no_grad():
                    latents, _, _ = encoder(images)

                specific_embed = specific(images)
                classifier_output = classifier(specific_embed, latents.detach())
                classifier_loss = classification_loss(classifier_output, labels)
                classifier_loss.backward()
                optimizer_specific.step()


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

