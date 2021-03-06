#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Convert a StyleGAN2 network stored in a .pkl file to ckpt files
#
#       Warning: it worked for my use case, not fully checked
#
import argparse
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import dnnlib
import dnnlib.tflib as tflib
from training import misc

def main():
    parser = argparse.ArgumentParser(
        description='Creates a ckpt from a pkl of a StyleGAN2 model.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ckpt_model_dir', help='The directory with the ckpt files', required=True)
    parser.add_argument('--input_pkl', help='A StyleGAN2 pkl', required=True)
    parser.add_argument('--prefix', default='')

    args = parser.parse_args()

    model_dir = args.ckpt_model_dir
    input_pkl = args.input_pkl
    prefix = args.prefix

    tflib.init_tf()
    with tf.Session() as sess:
        G, D, Gs = pickle.load(open(input_pkl, "rb"))
        G.print_layers(); D.print_layers(); Gs.print_layers()
        var_list = [v for v in tf.global_variables() if 'Dataset/' not in v.name]
        saver = tf.train.Saver(
        var_list=var_list,
        )
        saver.save(sess, model_dir+'/'+prefix+'model.ckpt')



if __name__ == '__main__':
    main()