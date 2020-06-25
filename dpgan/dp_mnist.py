# from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import tensorflow as tf
from dp.train import train
from models.gans import mnist
from utils.accounting import GaussianMomentsAccountant
from utils.data_utils import MNISTLoader
from utils.data_utils import MNISTLoader_aug
from utils.parsers import create_dp_parser
from utils.clippers import get_clipper
from utils.schedulers import get_scheduler
from dp.supervisors.basic_mnist import BasicSupervisorMNIST
from numpy import array
from utils.tf_DataSet import read_data_sets 
import os

if __name__ == "__main__":
    parser = create_dp_parser()
    parser.add_argument("--party", default=4, type=int, dest="party")
    parser.add_argument("--imbalanced", default=0, type=int, dest="imbalanced")
    parser.add_argument("--dim", default=64, type=int, dest="dim")
    parser.add_argument("--data-dir", default="./data/mnist_data", dest="data_dir")
    parser.add_argument("--learning-rate", default=2e-4, type=float, dest="learning_rate")
    parser.add_argument("--gen-learning-rate", default=2e-4, type=float, dest="gen_learning_rate")
    parser.add_argument("--adaptive-rate", dest="adaptive_rate", action="store_true")
    parser.add_argument("--sample-seed", dest="sample_seed", type=int, default=1024)
    parser.add_argument("--sample-ratio", dest="sample_ratio", type=float)
    parser.add_argument("--exclude-train", dest="exclude_train", action="store_true")
    parser.add_argument("--exclude-test", dest="exclude_test", action="store_true")

    config = parser.parse_args()
    config.dataset = "mnist"

    with open ("../"+config.dataset+"_party"+str(config.party)+'_imbalanced'+str(config.imbalanced), "r") as myfile:
        party_path=myfile.readlines()
    party_path=''.join(party_path)
    party_path = '../'+party_path
    nid=np.genfromtxt(party_path+'/nid',delimiter=',')
    nid = nid.astype(int) 
    print(nid)
    n_samples=np.genfromtxt(party_path+'/num_samples',delimiter=',')
    n_samples = n_samples.astype(int) 
    print(n_samples)
    config.image_dir=party_path+'/party'+str(nid)+'_samples'
    config.save_dir=party_path+'/party'+str(nid)+'_models'
    config.log_path=party_path+'/party'+str(nid)+'_log'

    train_label_indices=np.genfromtxt(party_path+'/party_indices',delimiter=',')
    train_label_indices = array( train_label_indices )  
    train_label_indices = train_label_indices.astype(int) 
    mnist_data = read_data_sets(train_dir=config.data_dir,validation_size=0,shard_index=train_label_indices[0:len(train_label_indices)])
    party_train = mnist_data.train
    party_data = party_train.images.reshape(party_train.images.shape[0], 1, 28, 28)
    party_labels = party_train.labels
    party_data_size = len(party_labels)
    print(party_data.shape)
    print(party_labels.shape)
    config.delta=1/(party_data_size*100)
    config.target_deltas=[1/(party_data_size*100)]

    np.random.seed()
    if config.enable_accounting:
        config.sigma = np.sqrt(2.0 * np.log(1.25 / config.delta)) / config.epsilon
        print("Now with new sigma: %.4f" % config.sigma)
    
    datagen = ImageDataGenerator(
    rotation_range=1,
    width_shift_range=0.01,
    height_shift_range=0.01)
    # fit parameters from data
    datagen.fit(party_data)
    # configure batch size and retrieve one batch of images
    expanded_images = []
    expanded_labels = []
    batches = 0
    #expand 100 times: 60000/100000 examples
    for X_batch, y_batch in datagen.flow(party_data, party_labels, batch_size=99):
        expanded_images.append(X_batch)
        expanded_labels.append(y_batch)
        batches += 1
        if batches >=party_data_size:
            break
    expanded_images.append(party_data)
    expanded_labels.append(party_labels)     
    expanded_images=np.vstack(expanded_images)
    expanded_images = np.concatenate(expanded_images, axis=0).reshape((-1, 28, 28, 1)).astype(np.float32)
    expanded_labels = np.concatenate(expanded_labels, axis=0)[:, None].astype(np.int64)
    indices = np.arange(expanded_images.shape[0])
    np.random.shuffle(indices)
    expanded_images = expanded_images[indices]
    expanded_labels = expanded_labels[indices]
    print(expanded_images.shape)
    print(expanded_labels.shape)

    if config.sample_ratio is not None:
        kwargs = {}
        gan_data_loader = MNISTLoader_aug(expanded_images, expanded_labels, 
                                  first=int(party_data_size *100 * (1 - config.sample_ratio)),
                                  seed=config.sample_seed
                                )
        sample_data_loader = MNISTLoader_aug(expanded_images, expanded_labels, 
                                  last=int(party_data_size *100 * config.sample_ratio),
                                  seed=config.sample_seed
                                )
    else:
        gan_data_loader = MNISTLoader(config.data_dir, include_train=not config.exclude_train,
                                  include_test=not config.exclude_test)

    if config.enable_accounting:
        accountant = GaussianMomentsAccountant(gan_data_loader.n, config.moment)
        if config.log_path:
            open(config.log_path, "w").close()
    else:
        accountant = None

    if config.adaptive_rate:
        lr = tf.placeholder(tf.float32, shape=())
    else:
        lr = config.learning_rate

    gen_optimizer = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=0.5, beta2=0.9)
    disc_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)

    clipper_ret = get_clipper(config.clipper, config)
    if isinstance(clipper_ret, tuple):
        clipper, sampler = clipper_ret
        sampler.set_data_loader(sample_data_loader)
        sampler.keep_memory = False
    else:
        clipper = clipper_ret
        sampler = None

    scheduler = get_scheduler(config.scheduler, config)
    def callback_before_train(_0, _1, _2):
        print(clipper.info())
    supervisor = BasicSupervisorMNIST(config, clipper, scheduler, sampler=sampler,
                                      callback_before_train=callback_before_train)
    if config.adaptive_rate:
        supervisor.put_key("lr", lr)
    print(gan_data_loader)
    train(config, gan_data_loader, mnist.generator_forward, mnist.discriminator_forward,
          gen_optimizer=gen_optimizer,
          disc_optimizer=disc_optimizer, accountant=accountant,
          supervisor=supervisor, n_samples=n_samples)
