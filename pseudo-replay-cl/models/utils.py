import models

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
