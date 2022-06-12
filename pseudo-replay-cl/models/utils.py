import utils
import copy
import torch
import models
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda import amp
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_specific_module(img_shape, config):
    if config['arch_specific'] == 'MLP':
        specific = models.mlp.Specific(img_shape=img_shape, specific_size=config['specific_size'])
    elif config['arch_specific'] == 'Conv':
        specific = models.conv.Specific(img_shape=img_shape, specific_size=config['specific_size'])

    specific = specific.to(DEVICE)

    return specific

def load_encoder(img_shape, config):
    if config['arch_encoder'] == 'IRCL':
        encoder = models.ircl.Encoder(img_shape=img_shape,
                                      n_hidden=300,
                                      latent_dim=config['latent_size'])
    elif config['arch_encoder'] == 'Conv':
        encoder = models.conv.Encoder(img_shape=img_shape,
                                      latent_dim=config['latent_size'])
    encoder = encoder.to(DEVICE)

    return encoder

def load_decoder(img_shape, n_classes, config):
    if config['arch_decoder'] == 'IRCL':
        decoder = models.ircl.Decoder(img_shape,
                                      300,
                                      config['latent_size'],
                                      n_classes)
    elif config['arch_decoder'] == 'Conv':
        decoder = models.conv.Decoder(img_shape,
                                      config['latent_size'],
                                      n_classes)

    decoder = decoder.to(DEVICE)

    return decoder

def load_classifier(n_classes, config):
    classifier = models.mlp.Classifier(invariant_size=config['latent_size'],
                                       specific_size=config['specific_size'],
                                       classification_n_hidden=40,
                                       n_classes=n_classes,
                                       softmax=config['softmax'])

    classifier = classifier.to(DEVICE)

    return classifier


def load_discriminator(img_shape, config):
    if config['arch_disc'] == 'MLP':
        discriminator = models.mlp.Discriminator(img_shape=img_shape)
    elif config['arch_disc'] == 'Conv':
        discriminator = models.conv.Discriminator(img_shape=img_shape)

    discriminator = discriminator.to(DEVICE)

    return discriminator

def train_vae(config, encoder, decoder, train_loader, val_loader=None, task_id=None):
    scaler = amp.GradScaler(enabled=config['use_amp'])
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=float(config['lr_autoencoder']))

    ### ------ Loss Functions ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    mlflow.log_params({'optimizer_cvae': 'Adam',
                       'loss_fn_pixelwise': 'MSE'})

    train_bar = tqdm(range(config['epochs_cvae']))

    best_valid_loss = 1e12
    for epoch in train_bar:
        encoder.train(); decoder.train()
        epoch_rec_loss, epoch_kl_loss, epoch_disc_loss = 0, 0, 0
        for images, labels, _ in train_loader:
            encoder.zero_grad(); decoder.zero_grad()

            labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
            images = images.to(DEVICE)

            with amp.autocast(enabled=config['use_amp']):
                latents, mu, var = encoder(images)
                recon_images = decoder(torch.cat([latents, labels_onehot],
                                                 dim=1))

                kl_loss = kl_divergence_loss(var, mu)/len(images)
                rec_loss = pixelwise_loss(recon_images, images)/len(images)
                cvae_loss = rec_loss + kl_loss

            scaler.scale(cvae_loss).backward()
            scaler.step(optimizer_cvae)
            scaler.update()

            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()

        log_str = lambda s: s if task_id is None else f'{s} Task {task_id}'
        rec_loss_str = log_str('Loss Reconstruction')
        cvae_loss_str = log_str('Loss CVAE')
        kl_loss_str = log_str('Loss KL')
        disc_loss_str = log_str('Loss Discriminator')

        cvae_total_loss = epoch_rec_loss + epoch_kl_loss

        mlflow.log_metrics({rec_loss_str: epoch_rec_loss/len(train_loader),
                            kl_loss_str: epoch_kl_loss/len(train_loader),
                            cvae_loss_str: (cvae_total_loss)/len(train_loader),
                            disc_loss_str: epoch_disc_loss},
                           step=epoch)

        if not config['use_validation_set']:
            continue

        encoder.eval(); decoder.eval()
        encoder.zero_grad(); decoder.zero_grad()

        with torch.inference_mode():
            valid_loss = 0
            for imgs, labels, _ in val_loader:
                labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
                imgs = imgs.to(DEVICE)

                with amp.autocast(enabled=config['use_amp']):
                    latents, mu, var = encoder(imgs)
                    recon_imgs = decoder(torch.cat([latents, labels_onehot],
                                                   dim=1))

                    kl_loss = kl_divergence_loss(var, mu)/len(imgs)
                    rec_loss = pixelwise_loss(recon_imgs, imgs)/len(imgs)

                kl_loss = scaler.scale(kl_loss)
                rec_loss = scaler.scale(rec_loss)

                cvae_loss = rec_loss + kl_loss
                valid_loss += cvae_loss.item()

        valid_loss_str = log_str('Loss VAE Validation')
        valid_loss /= len(val_loader)

        mlflow.log_metrics({valid_loss_str: valid_loss}, step=epoch)

        if valid_loss < best_valid_loss:
            early_stop_count = config['early_stop_count_vae']
            best_valid_loss = valid_loss
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
        else:
            early_stop_count -= 1

        if early_stop_count == 0:
            encoder = best_encoder
            decoder = best_decoder
            break


def train_vaegan(config, encoder, decoder, train_loader, val_loader, discriminator, task_id=None):
    scaler = amp.GradScaler(enabled=config['use_amp'])
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=float(config['lr_autoencoder']))
    if discriminator is not None:
        optimizer_discriminator = torch.optim.Adam(
                                        discriminator.parameters(),
                                        lr=float(config['lr_discriminator']),
                                        #betas=(0.5, 0.999)
                                        )
        discriminator_loss = torch.nn.BCEWithLogitsLoss()
        discriminator.train()
        real_label, recon_label = 1, 0

    best_valid_loss = 1e12
    ### ------ Loss Functions ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    mlflow.log_params({'optimizer_cvae': 'Adam',
                       'loss_fn_pixelwise': 'MSE'})

    train_bar = tqdm(range(config['epochs_cvae']))

    for epoch in train_bar:
        encoder.train(); decoder.train()
        epoch_rec_loss, epoch_kl_loss, epoch_disc_loss = 0, 0, 0
        for images, labels, _ in train_loader:
            encoder.zero_grad(); decoder.zero_grad()

            labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
            images = images.to(DEVICE)

            with amp.autocast(enabled=config['use_amp']):
                latents, mu, var = encoder(images)
                # Train Discriminator with all-real batch
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

                recon_images = decoder(torch.cat([latents, labels_onehot],
                                                 dim=1))

                # Train Discriminator with all-recon batch
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
            scaler.update()

            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()

            if discriminator is not None:
                epoch_disc_loss += lossG.item()

        log_str = lambda s: s if task_id is None else f'{s} Task {task_id}'
        rec_loss_str = log_str('Loss Reconstruction')
        cvae_loss_str = log_str('Loss CVAE')
        kl_loss_str = log_str('Loss KL')
        disc_loss_str = log_str('Loss Discriminator')

        cvae_total_loss = epoch_rec_loss + epoch_kl_loss

        if discriminator is not None:
            cvae_total_loss += epoch_disc_loss

        mlflow.log_metrics({rec_loss_str: epoch_rec_loss/len(train_loader),
                            kl_loss_str: epoch_kl_loss/len(train_loader),
                            cvae_loss_str: (cvae_total_loss)/len(train_loader),
                            disc_loss_str: epoch_disc_loss},
                           step=epoch)

        if not config['use_validation_set']:
            continue

        encoder.eval(); decoder.eval();
        encoder.zero_grad(); decoder.zero_grad();

        if discriminator is not None:
            discriminator.eval(); discriminator.zero_grad()

        with torch.inference_mode():
            valid_loss = 0
            for imgs, labels, _ in val_loader:
                labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
                imgs = imgs.to(DEVICE)

                with amp.autocast(enabled=config['use_amp']):
                    latents, mu, var = encoder(imgs)

                    if discriminator is not None:
                        disc_output = discriminator(imgs).view(-1)
                        real_labels = torch.ones(imgs.size(0)).type_as(imgs)
                        real_loss = discriminator_loss(disc_output, real_labels)

                        fake_labels = torch.zeros(imgs.size(0)).type_as(imgs)

                    recon_imgs = decoder(torch.cat([latents, labels_onehot],
                                                   dim=1))
                    if discriminator is not None:
                        disc_output = discriminator(recon_imgs).view(-1)
                        fake_loss = discriminator_loss(disc_output.detach(), fake_labels)

                    kl_loss = kl_divergence_loss(var, mu)/len(imgs)
                    rec_loss = pixelwise_loss(recon_imgs, imgs)/len(imgs)

                if discriminator is not None:
                    real_loss = scaler.scale(real_loss)
                    fake_loss = scaler.scale(fake_loss)

                kl_loss = scaler.scale(kl_loss)
                rec_loss = scaler.scale(rec_loss)

                cvae_loss = rec_loss + kl_loss
                if discriminator is not None:
                    cvae_loss += real_loss + fake_loss
                valid_loss += cvae_loss.item()

        valid_loss_str = log_str('Loss VAEGAN Validation')
        valid_loss /= len(val_loader)

        mlflow.log_metrics({valid_loss_str: valid_loss}, step=epoch)

        if valid_loss < best_valid_loss:
            early_stop_count = config['early_stop_count_vae']
            best_valid_loss = valid_loss
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
            best_discriminator = copy.deepcopy(discriminator)
        else:
            early_stop_count -= 1

        if early_stop_count == 0:
            encoder = best_encoder
            decoder = best_decoder
            discriminator = best_discriminator
            break


def train_classifier(config, encoder, specific, classifier, train_loader, val_loader, task_id=None):
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
    best_valid_loss = 1e12

    for epoch in train_bar:
        epoch_classifier_loss, epoch_acc = 0, 0
        specific.train(); classifier.train()
        classes_count = defaultdict(int)

        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            for cls in set(labels.tolist()):
                classes_count[cls] += len(labels[labels == cls])

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

        epoch_classifier_loss /= len(train_loader)

        log_str = lambda s: s if task_id is None else f'{s} Task {task_id}'
        cls_loss_str = log_str('Loss Classifier Train')

        mlflow.log_metrics({cls_loss_str: epoch_classifier_loss}, step=epoch)

        if not config['use_validation_set']:
            continue

        specific.eval(); classifier.eval()
        valid_loss = 0
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with amp.autocast(enabled=config['use_amp']):
                with torch.inference_mode():
                    latents, _, _ = encoder(imgs)
                    specific_embedding = specific(imgs)

                classifier_output = classifier(specific_embedding, latents.detach())
                classifier_loss = classification_loss(classifier_output, labels)/len(imgs)
            classifier_loss = scaler.scale(classifier_loss)
            valid_loss += classifier_loss.item()

        cls_valid_loss_str = log_str('Loss Classifier Validation')
        valid_loss /= len(val_loader)

        mlflow.log_metrics({cls_valid_loss_str: valid_loss}, step=epoch)

        if valid_loss < best_valid_loss:
            early_stop_count = config['early_stop_count_cls']
            best_valid_loss = valid_loss
            best_classifier = copy.deepcopy(classifier)
        else:
            early_stop_count -= 1

        if early_stop_count == 0:
            classifier = best_classifier
            break

    fig = plt.figure()
    plt.bar(list(classes_count.keys()), list(classes_count.values()))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for idx, key in enumerate(classes_count):
        ax.annotate(
            classes_count[key],
            xy=(idx - len(classes_count) * 0.035, classes_count[key] +
                (max(classes_count.values()) - min(classes_count.values())) *
                0.025),
            fontsize=10)

    plt.title(f'Task {task_id}')
    plt.savefig(dpi=300,
                fname=f'{task_plt_path}/classes_distribution.png',
                bbox_inches='tight')
    plt.close()




def test(encoder, specific, classifier, data_loader, use_amp=False, fname=None):
    encoder.eval(); classifier.eval(); specific.eval()

    with torch.no_grad():
        batch_acc = 0
        all_outputs = []
        all_labels = []
        for images, labels, _ in data_loader:
            images = images.to(DEVICE)
            with amp.autocast(enabled=use_amp):
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

