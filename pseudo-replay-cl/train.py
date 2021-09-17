import torch
import yaml
import data_utils
import ircl_models

def train(config):
    ### --------- ###
    ### Load Data ###
    ### --------- ###
    train_tasks, val_tasks, test_tasks = data_utils.load_tasks('MNIST',
                                                               val=True)
    data_shape = data_utils.get_task_data_shape(train_tasks)
    n_tasks = len(train_tasks)
    n_classes = data_utils.get_task_n_classes(train_tasks)

    ### -------------- ###
    ### Setting device ###
    ### -------------- ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ### ------------------- ###
    ### Loading IRCL Models ###
    ### ------------------- ###
    encoder, decoder, classifier = ircl_models.load_all(data_shape,
                                                        n_classes,
                                                        device)
    ### -_-------- ###
    ### Optimizers ###
    ### -_-------- ###
    optimizer_cvae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=config['learning_rate'])
    optimizer_C = torch.optim.Adam(classifier.parameters(),
                                   lr=config['learning_rate']/50)

def load_config(file):
    config = None
    with open(file, 'r') as reader:
        config = yaml.load(reader)

    return config

if __name__ == '__main__':
    config = load_config('config.yaml')
    train(config)
