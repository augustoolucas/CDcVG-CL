import utils
import torch
import models
import mlflow
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda import amp
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

    for epoch in train_bar:
        encoder.train(); decoder.train()
        epoch_rec_loss, epoch_kl_loss, epoch_disc_loss = 0, 0, 0
        for images, labels in train_loader:
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


def train_vaegan(config, encoder, decoder, train_loader, val_loader=None, discriminator=None, task_id=None):
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

    ### ------ Loss Functions ------ ###
    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    kl_divergence_loss = lambda var, mu: 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)

    mlflow.log_params({'optimizer_cvae': 'Adam',
                       'loss_fn_pixelwise': 'MSE'})

    train_bar = tqdm(range(config['epochs_cvae']))

    for epoch in train_bar:
        encoder.train(); decoder.train()
        epoch_rec_loss, epoch_kl_loss, epoch_disc_loss = 0, 0, 0
        ssim_epoch = 0
        for images, labels in train_loader:
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

                """
                for img, recon_img in zip(images, recon_images):
                    img = img_as_float(img.squeeze().detach().cpu().numpy())
                    recon_img = img_as_float(recon_img.squeeze().detach().cpu().numpy())
                    ssim_epoch += ssim(img, recon_img,
                                       data_range=recon_img.max() - recon_img.min())
                ssim_epoch /= len(images)
                """

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
        ssim_str = log_str('SSIM')

        cvae_total_loss = epoch_rec_loss + epoch_kl_loss

        if discriminator is not None:
            cvae_total_loss += epoch_disc_loss

        mlflow.log_metrics({rec_loss_str: epoch_rec_loss/len(train_loader),
                            kl_loss_str: epoch_kl_loss/len(train_loader),
                            cvae_loss_str: (cvae_total_loss)/len(train_loader),
                            disc_loss_str: epoch_disc_loss},
                            #ssim_str: ssim_epoch/len(train_loader)},
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

        cls_loss_str = 'Loss Classifier' if task_id is None else f'Loss Classifier Task {task_id}' 
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


def test(encoder, decoder, specific, classifier, data_loader, use_amp=False, fname=None, log_metrics=False):
    encoder.eval(); classifier.eval(); specific.eval(); decoder.eval()

    pixelwise_loss = torch.nn.MSELoss(reduction='sum')
    with torch.no_grad():
        batch_acc = 0
        all_outputs = []
        all_labels = []
        ssim_test = 0
        n_imgs = 0
        rec_loss = 0
        for images, labels in data_loader:
            n_imgs += len(images)
            images = images.to(DEVICE)
            labels_onehot = utils.data.onehot_encoder(labels, 10).to(DEVICE)
            with amp.autocast(enabled=use_amp):
                latents, mu, var = encoder(images)
                recon_images = decoder(torch.cat([latents, labels_onehot], dim=1))
                rec_loss = pixelwise_loss(recon_images, images)/len(images)

                specific_output = specific(images)
                classifier_output = classifier(specific_output, latents.detach())

            for img, recon_img in zip(images, recon_images):
                img = img_as_float(img.squeeze().detach().cpu().numpy())
                recon_img = img_as_float(recon_img.squeeze().detach().cpu().numpy())
                ssim_test += ssim(img, recon_img,
                                  data_range=recon_img.max() - recon_img.min())


            classifier_output = torch.argmax(classifier_output, dim=1).tolist()
            batch_acc += accuracy_score(classifier_output, labels.tolist())

            if fname:
                all_outputs.extend(classifier_output)
                all_labels.extend(labels.tolist())

        ssim_test /= n_imgs
        rec_loss /= len(data_loader)
        if log_metrics:
            mlflow.log_metric('SSIM Test', ssim_test)
            mlflow.log_metric('Reconstruction Loss Test', rec_loss.item())

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

