"""
"""
from __future__ import division

import math
import os
import warnings

import numpy as np
import scipy.io.wavfile
import scipy.signal
from picklable_itertools import cycle
from picklable_itertools.extras import partition_all
from tqdm import tqdm

class DataSet(object):
    def __init__(self, configuration, fragment_length, num_gpus=1, train_rank=0):
        self.fragment_length = fragment_length
        self.data_dir = configuration.get('model', 'data_dir')
        self.data_dir_structure = configuration.get('model', 'data_dir_structure')
        self.model_dir = configuration.get('model', 'model_dir')
        self.sample_rate = configuration.getint('model', 'sample_rate')
        self.debug = configuration.getint('model', 'debug')
        # Training Configuration
        self.max_epoch = configuration.getint('training', 'max_epoch')
        self.test_factor = configuration.getfloat('training', 'test_factor')
        self.batch_size = configuration.getint('training', 'batch_size')
        self.output_bins = configuration.getint('training', 'output_bins')
        self.filters = configuration.getint('training', 'filters')
        self.dilation_depth = configuration.getint('training', 'dilation_depth')
        self.stacks = configuration.getint('training', 'stacks')
        self.use_bias = configuration.getboolean('training', 'use_bias')
        self.use_ulaw = configuration.getboolean('training', 'use_ulaw')
        self.res_l2 = configuration.getfloat('training', 'res_l2')
        self.final_l2 = configuration.getfloat('training', 'final_l2')
        self.initial_fragment_length = configuration.getint('training', 'initial_fragment_length')
        self.fragment_stride = configuration.getint('training', 'fragment_stride')
        self.use_skip_connections = configuration.getboolean('training', 'use_skip_connections')
        self.learn_all_outputs = configuration.getboolean('training', 'learn_all_outputs')
        self.random_train_batches = configuration.getboolean('training', 'random_train_batches')
        self.randomize_batch_order = configuration.getboolean('training', 'randomize_batch_order')
        self.train_only_in_receptive_field = configuration.getboolean('training', 'train_only_in_receptive_field')
        self.train_with_soft_targets = configuration.getboolean('training', 'train_with_soft_targets')
        self.soft_target_stdev = configuration.getfloat('training', 'soft_target_stdev')
        self.optimizer = configuration.get('training', 'optimizer')
        self.early_stopping_patience = configuration.getint('training', 'early_stopping_patience')
        # Prediction Configuration
        self.predict_length = configuration.getint('prediction', 'predict_length')
        self.sample_argmax = configuration.getboolean('prediction', 'sample_argmax')
        self.sample_temperature = configuration.getfloat('prediction', 'sample_temperature')
        self.predict_initial_input = configuration.get('prediction', 'initial_input')
        self.train_rank = train_rank
        self.num_gpus = num_gpus

    def one_hot(self, x):
        return np.eye(256, dtype='uint8')[x.astype('uint8')]

    def fragment_indices(self, full_sequences):
        for seq_i, sequence in enumerate(full_sequences):
            # range_values = np.linspace(np.iinfo(sequence.dtype).min, np.iinfo(sequence.dtype).max, nb_output_bins)
            # digitized = np.digitize(sequence, range_values).astype('uint8')
            for i in xrange(0, sequence.shape[0] - self.fragment_length, self.fragment_stride):
                yield seq_i, i

    def select_generator(self, set_name, full_sequences, rnd):
        if self.random_train_batches and set_name == 'train':
            bg = self.random_batch_generator
        else:
            bg = self.batch_generator
        return bg(full_sequences, rnd)

    def batch_generator(self, full_sequences, rnd):
        indices = list(self.fragment_indices(full_sequences))
        if self.randomize_batch_order:
            rnd.shuffle(indices)

        batches_parted = [batch for batch in partition_all(self.batch_size, indices)]
        start_index = len(batches_parted) // self.num_gpus * self.train_rank
        batches_gpu = batches_parted[start_index:]
        batches = cycle(batches_gpu)
        for batch in batches:
            if len(batch) < self.batch_size:
                continue
            yield np.array(
                [self.one_hot(full_sequences[e[0]][e[1]:e[1] + self.fragment_length]) for e in batch], dtype='uint8'), np.array(
                [self.one_hot(full_sequences[e[0]][e[1] + 1:e[1] + self.fragment_length + 1]) for e in batch], dtype='uint8')

    def random_batch_generator(self, full_sequences, rnd):
        lengths = [x.shape[0] for x in full_sequences]
        nb_sequences = len(full_sequences)
        while True:
            sequence_indices = rnd.randint(0, nb_sequences, self.batch_size)
            batch_inputs = []
            batch_outputs = []
            for i, seq_i in enumerate(sequence_indices):
                l = lengths[seq_i]
                offset = np.squeeze(rnd.randint(0, l - fragment_length, 1))
                batch_inputs.append(full_sequences[seq_i][offset:offset + self.fragment_length])
                batch_outputs.append(full_sequences[seq_i][offset + 1:offset + self.fragment_length + 1])
            yield self.one_hot(np.array(batch_inputs, dtype='uint8')), self.one_hot(np.array(batch_outputs, dtype='uint8'))

    def generators(self, rnd):
        fragment_generators = {}
        nb_examples = {}
        for set_name in ['train', 'test']:
            set_dirname = os.path.join(self.data_dir, set_name)
            full_sequences = self._load_set(set_dirname)
            fragment_generators[set_name] = self.select_generator(set_name, full_sequences, rnd)
            nb_examples[set_name] = int(sum( [len(xrange(0, x.shape[0] - self.fragment_length, self.fragment_stride)) for x in full_sequences]) / self.batch_size) * self.batch_size
        return fragment_generators, nb_examples

    def _load_set(self, set_dirname):
        ulaw_str = '_ulaw' if self.use_ulaw else ''
        cache_fn = os.path.join(set_dirname, 'processed_%d%s.npy' % (self.sample_rate, ulaw_str))
        if os.path.isfile(cache_fn):
            full_sequences = np.load(cache_fn)
        else:
            file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith('.wav')]
            full_sequences = []
            for fn in tqdm(file_names):
                sequence = self.process_wav(os.path.join(set_dirname, fn))
                full_sequences.append(sequence)
            np.save(cache_fn, full_sequences)
        return full_sequences

    def process_wav(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            channels = scipy.io.wavfile.read(filename)
        file_sample_rate, audio = channels
        audio = self.ensure_mono(audio)
        audio = self.wav_to_float(audio)
        if self.use_ulaw:
            audio = self.ulaw(audio)
        audio = self.ensure_sample_rate(file_sample_rate, audio)
        audio = self.float_to_uint8(audio)
        return audio

    def ulaw(self, x, u=255):
        x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
        return x

    def float_to_uint8(self, x):
        x += 1.
        x /= 2.
        uint8_max_value = np.iinfo('uint8').max
        x *= uint8_max_value
        x = x.astype('uint8')
        return x

    def wav_to_float(self, x):
        try:
            max_value = np.iinfo(x.dtype).max
            min_value = np.iinfo(x.dtype).min
        except:
            max_value = np.finfo(x.dtype).max
            min_value = np.iinfo(x.dtype).min
        x = x.astype('float32', casting='safe')
        x -= min_value
        x /= ((max_value - min_value) / 2.)
        x -= 1.
        return x

    def ulaw2lin(self, x, u=255.):
        max_value = np.iinfo('uint8').max
        min_value = np.iinfo('uint8').min
        x = x.astype('float32', casting='safe')
        x -= min_value
        x /= ((max_value - min_value) / 2.)
        x -= 1.
        x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
        x = self.float_to_uint8(x)
        return x

    def ensure_sample_rate(self, file_sample_rate, mono_audio):
        if file_sample_rate != self.sample_rate:
            mono_audio = scipy.signal.resample_poly(mono_audio, self.sample_rate, file_sample_rate)
        return mono_audio

    def ensure_mono(self, raw_audio):
        """
        Just use first channel.
        """
        if raw_audio.ndim == 2:
            raw_audio = raw_audio[:, 0]
        return raw_audio

