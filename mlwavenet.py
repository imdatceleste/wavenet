# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import datetime
import json
import os
import re
import wave
import keras.backend as K
import numpy as np
import scipy.io.wavfile
import scipy.signal
import getopt
import ConfigParser
import codecs
from keras import layers
from keras import metrics
import keras.utils as KU
from keras import objectives
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.engine import Input
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras import models
from tqdm import tqdm
from dataset import DataSet
import dataset
from wavenet_utils import CausalDilatedConv1D, categorical_mean_squared_error
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:C:r:Rl:me:', ['--config', '--CMD', '--resume', '--restart', '--length', '--mgpu', '--epoch'])
    except getopt.GetoptError:
        print_usage()

    config_file = None
    command = 'train'
    resume_training = True
    resume_epoch = None
    predict_length = None
    multi_gpu = False
    epoch = None

    for opt, arg in opts:
        if opt in ('-c', '--config'):
            config_file = arg
        elif opt in ('-C', '--CMD'):
            command = arg
        elif opt in ('-r', '--resume'):
            resume_epoch = int(arg)
        elif opt in ('-R', '--restart'):
            resume_training = False
        elif opt in ('-l', '--length'):
            predict_length = int(arg)
        elif opt in ('-m', '--mgpu'):
            multi_gpu = True
        elif opt in ('-e', '--epoch'):
            epoch = int(arg)

    if multi_gpu:
        import tensorflow as tf
        import horovod.keras as hvd

        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print('hdv.local_rank: ', hvd.local_rank())
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))

"""
Keras2 based WaveNet

Based originally on Bas Veeling's implementation at: https://github.com/basveeling/wavenet/
which is (c) Bas Veeling

Copyright (c) MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Updated: 2017-12-07 - 2017-12-18, Imdat Solak (ISO)
         - Ported to Keras2
         - Removed dependency on Theano (now fully supports Tensorflow OR other Keras-backend)
         - Removed dependency on Sacred: Now you can use configuration files and have some
           more reliability in training + testing
         - Added support for resuming training at any epoch
         - Added support for Multi-GPU Training using Horovod & OpenMPI
         - Reduced RAM usage when predicting:
              Initially it was using about 250 GiB(!) for 60 seconds @ 22KHz.
              Now, it only uses about 2GiB of RAM regardless of prediction seconds @ 22KHz;
              My assumption is that it will use about 4GiB RAM at 44KHz :-)
         - Moved everything into object-oriented structure
         - Removed support for VCTK (sorry, but I want this to generate music, not speech;
           we have another one for Speech Synthesis :-)
         - Increased performance slightly
"""

class MLWaveNet(object):
    def __init__(self, config_file, resume_training=True, resume_epoch=None, predict_length=None, multi_gpu=False):
        self.config = ConfigParser.ConfigParser(allow_no_value=True)
        try:
            self.config.readfp(open(config_file))
        except:
            print('Could not read configuration file {} - exiting.'.format(config_file))
            sys.exit(1)
        # Get General Configuration
        self.train_multi_gpu = multi_gpu
        self.resume_training = resume_training
        self.resume_epoch = resume_epoch
        self.keras_verbose = self.config.getint('general', 'keras_verbose')
        self.seed = self.config.getint('general', 'seed')
        if self.seed is None:
            self.seed = 42
        # Get Model Configuration
        self.data_dir = self.config.get('model', 'data_dir')
        self.data_dir_structure = self.config.get('model', 'data_dir_structure')
        self.model_dir = self.config.get('model', 'model_dir')
        if len(self.model_dir) == 0:
            self.model_dir = None
        self.sample_rate = self.config.getint('model', 'sample_rate')
        self.debug = self.config.getint('model', 'debug')
        # Training Configuration
        self.max_epoch = self.config.getint('training', 'max_epoch')
        self.test_factor = self.config.getfloat('training', 'test_factor')
        self.batch_size = self.config.getint('training', 'batch_size')
        self.output_bins = self.config.getint('training', 'output_bins')
        self.filters = self.config.getint('training', 'filters')
        self.dilation_depth = self.config.getint('training', 'dilation_depth')
        self.stacks = self.config.getint('training', 'stacks')
        self.use_bias = self.config.getboolean('training', 'use_bias')
        self.use_ulaw = self.config.getboolean('training', 'use_ulaw')
        self.res_l2 = self.config.getfloat('training', 'res_l2')
        self.final_l2 = self.config.getfloat('training', 'final_l2')
        self.initial_fragment_length = self.config.getint('training', 'initial_fragment_length')
        self.fragment_stride = self.config.getint('training', 'fragment_stride')
        self.use_skip_connections = self.config.getboolean('training', 'use_skip_connections')
        self.learn_all_outputs = self.config.getboolean('training', 'learn_all_outputs')
        self.random_train_batches = self.config.getboolean('training', 'random_train_batches')
        self.randomize_batch_order = self.config.getboolean('training', 'randomize_batch_order')
        self.train_only_in_receptive_field = self.config.getboolean('training', 'train_only_in_receptive_field')
        self.train_with_soft_targets = self.config.getboolean('training', 'train_with_soft_targets')
        self.soft_target_stdev = self.config.getfloat('training', 'soft_target_stdev')
        self.optimizer = self.config.get('training', 'optimizer')
        self.early_stopping_patience = self.config.getint('training', 'early_stopping_patience')
        # Prediction Configuration
        self.predict_length = self.config.getfloat('prediction', 'predict_length')
        # Let's allow the user to overwrite the length via cmd-line, it is more practical :-)
        if predict_length is not None:
            self.predict_length = predict_length
        self.sample_argmax = self.config.getboolean('prediction', 'sample_argmax')
        self.sample_temperature = self.config.getfloat('prediction', 'sample_temperature')
        if self.sample_temperature < 0.001:
            self.sample_temperature = None
        self.predict_initial_input = self.config.get('prediction', 'initial_input')
        if len(self.predict_initial_input) == 0:
            self.predict_initial_input = None
        self.predict_use_softmax_as_input = self.config.getboolean('prediction', 'use_softmax_as_input')
        self.sample_seed = self.seed
        np.random.seed(self.seed)
        self.rnd = np.random.RandomState(self.seed)

        self.fragment_length = self.initial_fragment_length + self._compute_receptive_field2(self.sample_rate, self.dilation_depth, self.stacks)[0]
        # Additional Settings
        self.num_gpus = 1
        self.train_rank = 0
        if self.train_multi_gpu:
            self.train_rank = hvd.rank()
            self.num_gpus = hvd.size()
        print('rank = {}, num_gpu={}'.format(self.train_rank, self.num_gpus))
        self.dataset = DataSet(self.config, self.fragment_length, self.num_gpus, self.train_rank)

    # ################################################################################################## 
    # Various Methods
    # ################################################################################################## 
    def _compute_receptive_field(self):
        return self._compute_receptive_field2(self.sample_rate, self.dilation_depth, self.stacks)

    def _compute_receptive_field2(self, sample_rate, dilation_depth, stacks):
        receptive_field = stacks * (2 ** dilation_depth * 2) - (stacks - 1)
        receptive_field_ms = (receptive_field * 1000) / sample_rate
        return receptive_field, receptive_field_ms

    def _skip_out_of_receptive_field(self, func):
        # TODO: consider using keras masking for this?
        receptive_field, _ = self._compute_receptive_field()

        def wrapper(y_true, y_pred):
            y_true = y_true[:, receptive_field - 1:, :]
            y_pred = y_pred[:, receptive_field - 1:, :]
            return func(y_true, y_pred)

        wrapper.__name__ = func.__name__
        return wrapper

    # ################################################################################################## 
    # Data Generators
    # ################################################################################################## 
    def _get_generators(self):
        if self.data_dir_structure == 'flat':
            return self.dataset.generators(self.rnd)
        # elif data_dir_structure == 'vctk':
        #     return dataset.generators_vctk(self.data_dir, self.sample_rate, self.fragment_length, self.batch_size, self.fragment_stride, self.output_bins, self.learn_all_outputs, self.use_ulaw, self.test_factor, self.randomize_batch_order, self.rnd, self.random_train_batches)
        else:
            raise ValueError('data_dir_structure must be "flat" or "vctk", is %s' % self.data_dir_structure)

    # ################################################################################################## 
    # Building the model
    # ################################################################################################## 
    def _build_model_residual_block(self, x, i, s):
        original_x = x
        # TODO: initalization, regularization?
        tanh_out = CausalDilatedConv1D(self.filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True, bias=self.use_bias, name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh', W_regularizer=l2(self.res_l2))(x)
        sigm_out = CausalDilatedConv1D(self.filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True, bias=self.use_bias, name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid', W_regularizer=l2(self.res_l2))(x)
        x = layers.Multiply()([tanh_out, sigm_out])

        res_x = layers.Conv1D(self.filters, 1, padding='same', use_bias=self.use_bias, kernel_regularizer=l2(self.res_l2))(x)
        skip_x = layers.Conv1D(self.filters, 1, padding='same', use_bias=self.use_bias, kernel_regularizer=l2(self.res_l2))(x)
        res_x = layers.Add()([original_x, res_x])
        return res_x, skip_x

    def _build_model(self):
        input_shape = Input(shape=(self.fragment_length, self.output_bins), name='input_part')
        out = input_shape
        skip_connections = []
        out = CausalDilatedConv1D(self.filters, 2, atrous_rate=1, border_mode='valid', causal=True, name='initial_causal_conv')(out)
        for s in range(self.stacks):
            for i in range(0, self.dilation_depth + 1):
                out, skip_out = self._build_model_residual_block(out, i, s)
                skip_connections.append(skip_out)

        if self.use_skip_connections:
            out = layers.Add()(skip_connections)
        out = layers.Activation('relu')(out)
        out = layers.Conv1D(self.output_bins, 1, padding='same', kernel_regularizer=l2(self.final_l2))(out)
        out = layers.Activation('relu')(out)
        out = layers.Conv1D(self.output_bins, 1, padding='same')(out)
        if not self.learn_all_outputs:
            raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
            out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(out)  # Based on gif in deepmind blog: take last output?

        out = layers.Activation('softmax', name="output_softmax")(out)
        model = Model(input_shape, out)
        self.receptive_field, self.receptive_field_ms = self._compute_receptive_field()
        return model

    # ################################################################################################## 
    # Loading a Checkpoint (training + prediction)
    # ################################################################################################## 
    def _get_checkpoint_file(self, checkpoint_no=None):
        if checkpoint_no is not None:
            checkpoint_file  = os.path.join(self.model_dir, 'checkpoints', 'checkpoint.{:05d}.hdf5'.format(checkpoint_no))
        else:
            history_file = os.path.join(self.model_dir, 'history.csv')
            print('Reading history... {}'.format(history_file))
            try:
                history = codecs.open(history_file, 'r', 'utf-8').readlines()
                checkpoint_no = int(history[-1].strip().split(',')[0]) + 1
            except:
                checkpoint_no = 0
            checkpoint_file  = os.path.join(self.model_dir, 'checkpoints', 'checkpoint.{:05d}.hdf5'.format(checkpoint_no))
        return checkpoint_file, checkpoint_no

    def _load_model_weights(self, weights_file=None, checkpoint_no=None):
        if weights_file is None:
            self.checkpoint_file, self.resume_epoch = self._get_checkpoint_file(self.resume_epoch)
        else:
            self.checkpoint_file = weights_file
            self.resume_epoch = checkpoint_no
        if self.resume_epoch > 0 and os.path.exists(self.checkpoint_file):
            self.model.load_weights(self.checkpoint_file)
            return True, self.resume_epoch
        else:
            return False, 0
    
    # ################################################################################################## 
    # Training, including resuming training
    # ################################################################################################## 
    def _make_soft(self, y_true):
        self.receptive_field, _ = self._compute_receptive_field()
        n_outputs = self.fragment_length - self.receptive_field + 1
        # Make a gaussian kernel.
        kernel_v = scipy.signal.gaussian(9, std=self.soft_target_stdev)
        # print(kernel_v)
        kernel_v = np.reshape(kernel_v, [1, 1, -1, 1])
        kernel = K.variable(kernel_v)
        # y_true: [batch, timesteps, input_dim]
        y_true = K.reshape(y_true, (-1, 1, self.output_bins, 1))  # Same filter for all output; combine with batch.
        # y_true: [batch*timesteps, n_channels=1, input_dim, dummy]
        y_true = K.conv2d(y_true, kernel, border_mode='same')
        y_true = K.reshape(y_true, (-1, n_outputs, self.output_bins))  # Same filter for all output; combine with batch.
        # y_true: [batch, timesteps, input_dim]
        y_true /= K.sum(y_true, axis=-1, keepdims=True)
        return y_true

    def _make_targets_soft(self, func):
        """Turns one-hot into gaussian distributed."""
        def wrapper(y_true, y_pred):
            y_true = self._make_soft(y_true)
            y_pred = y_pred
            return func(y_true, y_pred)

        wrapper.__name__ = func.__name__
        return wrapper

    def _make_optimizer(self):
        section_name = 'optimizer-' + self.optimizer
        lr = self.config.getfloat(section_name, 'lr')
        decay = self.config.getfloat(section_name, 'decay')
        if self.optimizer == 'sgd':
            momentum = self.config.getfloat(section_name, 'momentum')
            nesterov = self.config.getboolean(section_name, 'nesterov')
            optim = SGD(lr, momentum, decay, nesterov)
        elif self.optimizer == 'adam':
            epsilon = self.config.getfloat(section_name, 'epsilon')
            optim = Adam(lr=lr, decay=decay, epsilon=epsilon)
        else:
            raise ValueError('Invalid config for optimizer.optimizer: ' + self.optimizer)
        return optim

    def train(self, resume_training=True, resume_epoch=None):
        if self.model_dir is None or self.model_dir == '':
            self.model_dir = os.path.join('models', datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S'))
            self.load_model = False
            self.resume_training = False 
        else:
            self.load_model = True

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        self.data_generators, self.nb_examples = self._get_generators()

        self.model = self._build_model()
        # _log.info(model.summary())

        loss = objectives.categorical_crossentropy
        all_metrics = [
            metrics.categorical_accuracy,
            categorical_mean_squared_error
        ]

        if self.train_with_soft_targets:
            loss = self._make_targets_soft(loss)
        if self.train_only_in_receptive_field:
            loss = self._skip_out_of_receptive_field(loss)
            all_metrics = [self._skip_out_of_receptive_field(m) for m in all_metrics]

        optim = self._make_optimizer()
        if self.train_multi_gpu:
            optim = hvd.DistributedOptimizer(optim)
        self.model.compile(optimizer=optim, loss=loss, metrics=all_metrics)

        self.initial_epoch = 0
        if self.resume_training:
            _, self.initial_epoch = self._load_model_weights()

        # TODO: Consider gradient weighting making last outputs more important.

        if self.train_multi_gpu:
            callbacks = [
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                hvd.callbacks.MetricAverageCallback(),
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1)
            ]
        else:
            callbacks = []

        callbacks.extend([
            ReduceLROnPlateau(patience=self.early_stopping_patience / 2, cooldown=self.early_stopping_patience / 4, verbose=1),
            EarlyStopping(patience=self.early_stopping_patience, verbose=1)
        ])
        if self.train_rank == 0:
            callbacks.extend([
                        ModelCheckpoint(os.path.join(self.checkpoint_dir, 'checkpoint.{epoch:05d}.hdf5'), save_best_only=False),
                        CSVLogger(os.path.join(self.model_dir, 'history.csv'), append=True)
                    ])
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        keras_verbose = self.keras_verbose
        if self.train_rank > 0:
            keras_verbose = 0
        else:
            print('Starting Training...')
        self.model.fit_generator(self.data_generators['train'],
                            self.nb_examples['train'] // self.num_gpus,
                            initial_epoch=self.initial_epoch,
                            epochs=self.max_epoch,
                            validation_data=self.data_generators['test'],
                            validation_steps=self.nb_examples['test'] // self.num_gpus,
                            callbacks=callbacks,
                            verbose=keras_verbose)

    # ################################################################################################## 
    # Prediction
    # ################################################################################################## 
    def _make_sample_stream(self, sample_filename):
        sample_file = wave.open(sample_filename, mode='w')
        sample_file.setnchannels(1)
        sample_file.setframerate(self.sample_rate)
        sample_file.setsampwidth(1)
        return sample_file

    def _softmax(self, x):
        x = np.log(x) / self.sample_temperature
        e_x = np.exp(x - np.max(x, axis=-1))
        return e_x / np.sum(e_x, axis=-1)

    def _draw_sample(self, output_dist):
        if self.sample_argmax:
            output_dist = np.eye(256)[np.argmax(output_dist, axis=-1)]
        else:
            if self.sample_temperature is not None:
                output_dist = self._softmax(output_dist)
            output_dist = output_dist / np.sum(output_dist + 1e-7)
            output_dist = self.rnd.multinomial(1, output_dist)
        return output_dist

    def _make_sample_name(self, epoch):
        sample_str = ''
        if self.predict_use_softmax_as_input:
            sample_str += '_soft-in'
        if self.sample_argmax:
            sample_str += '_argmax'
        else:
            sample_str += '_sample'
            if self.sample_temperature:
                sample_str += '-temp-%s' % self.sample_temperature
        sample_name = 'sample_epoch-%05d_%02ds_%s_seed-%d.wav' % (epoch, int(self.predict_length), sample_str, self.sample_seed)
        return sample_name

    def _write_samples(self, sample_file, out_val):
        s = np.argmax(out_val, axis=-1).astype('uint8')
        if self.use_ulaw:
            s = self.dataset.ulaw2lin(s)
        s = bytearray(list(s))
        sample_file.writeframes(s)
        sample_file._file.flush()

    def predict(self, epoch=None):
        self.fragment_length = self._compute_receptive_field()[0]
        last_checkpoint_file, epoch = self._get_checkpoint_file(epoch)

        sample_dir = os.path.join(self.model_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        sample_name = self._make_sample_name(epoch)
        sample_filename = os.path.join(sample_dir, sample_name)
        sample_stream = self._make_sample_stream(sample_filename)

        self.model = self._build_model()
        print('Loading model: {}'.format(last_checkpoint_file))
        model_is_loaded, _ = self._load_model_weights(last_checkpoint_file, epoch)
        if model_is_loaded:
            # KU.print_summary(self.model, line_length=200)
            # KU.plot_model(self.model, to_file='plot.png', show_shapes=True)

            if self.predict_initial_input is None or self.predict_initial_input == 'random':
                # Random initial data
                print('RANDOM initial input')
                outputs = list(self.dataset.one_hot(np.random.randn(self.fragment_length) + self.output_bins / 2))
            elif self.predict_initial_input == 'zero':
                # zero initial input
                print('ZERO initial input')
                outputs = list(self.dataset.one_hot(np.zeros(self.fragment_length) + self.output_bins / 2))
            elif self.predict_initial_input != 'test':
                # Take from provided file
                print('Initial Input is -{}-'.format(self.predict_initial_input))
                outputs = list(self.dataset.one_hot(np.random.randn(self.fragment_length) + self.output_bins / 2))
                wav = self.dataset.process_wav(self.predict_initial_input)
                outputs = list(self.dataset.one_hot(wav[0:self.fragment_length]))
            else:
                # Take from test dataset
                print('Initial Input is test -{}-'.format(self.predict_initial_input))
                self.data_generators, _ = self._get_generators()
                outputs = list(self.data_generators['test'].next()[0][-1])

            for i in tqdm(xrange(int(self.sample_rate * self.predict_length))):
                prediction_seed = np.expand_dims(np.array(outputs[:self.fragment_length]), 0)
                output = self.model.predict(prediction_seed)
                output_dist = output[0][-1]
                output_val = self._draw_sample(output_dist)
                if self.predict_use_softmax_as_input:
                    outputs.append(output_dist)
                else:
                    outputs.append(output_val)
                self._write_samples(sample_stream, [output_val])
                del outputs[0]
            sample_stream.close()
        else:
            print('Could not load model {} -- exiting.'.format(last_checkpoint_file))


def print_usage():
    print('Usage:')
    print('\tpython mlwavenet.py -c <config-file> [-C <train|test|predict>] [-r <resume-epoch>] [-R] [-l predict_length] [-e epoch]')
    sys.exit(1)


if __name__ == '__main__': 
    if config_file is None:
        print_usage()

    wavenet = MLWaveNet(config_file, resume_training, resume_epoch, predict_length, multi_gpu)
    if command == 'train':
        wavenet.train()
    elif command == 'test':
        pass
    elif command == 'predict':
        wavenet.predict(epoch)
    else:
        print('Unknown command <{}>.'.format(command))
        print_usage()
