import os
import tarfile
import argparse

import pandas as pd
from tqdm import tqdm
import tensorflow as tf


class PersonIdAudio:
    def __init__(self, audio_content, sr, verbose=0):
        """ Constructor

        Arguments:

        audio_content: dictionary containing
            person_id as keys and a
            list of mp3-encoded samples

        sr: sampling rate
        """
        self.audio_content = audio_content
        person_ids = audio_content.keys()
        labels = range(len(person_ids))
        self.id_to_label = dict(zip(person_ids, labels))
        self.n_audios = sum([len(audio_content[x]) for x in audio_content])
        self.verbose = verbose

    def get_tf_dataset(self):
        """ Retrieves a tf.records.dataset that
            produces pairs of (audio_data, user_id)
        """
        audio_signature = (
            tf.TensorSpec(shape=(None), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        audio_dataset = tf.data.Dataset.from_generator(
             self.gen_audios,
             output_signature=audio_signature
        )
        return audio_dataset

    def gen_audios(self):
        """ Generate audios and id's
        Leave the shuffling part to tf dataset
        """
        if self.verbose == 0:
            iterator = self.audio_content
        else:
            iterator = tqdm(self.audio_content)

        for person_id in iterator:
            person_label = self.id_to_label[person_id]
            for item in self.audio_content[person_id]:
                yield item, person_label

    def save_tfrecords_file(self, output_file, compression_type='GZIP'):
        """ Saves this dataset in tfrecords format

        Arguments:

        output_file - file to write to
        compression_type - compression to use

        Returns: output_file
        """
        if '.' not in output_file:
            output_file = output_file + '.tfrecords'
            if compression_type is not None:
                output_file = output_file + '.' + compression_type

        audio_dataset = self.get_tf_dataset()
        audio_dataset = audio_dataset.map(
            PersonIdAudio._serialize_for_tfrecords
        )
        writer = tf.data.experimental.TFRecordWriter(
            output_file,
            compression_type=compression_type
        )
        writer.write(audio_dataset)
        return output_file

    # static functions
    def _serialize_for_tfrecords(audio_contents, label):
        # audio_contents is already a string
        s_label = tf.io.serialize_tensor(label)
        stacked_tensor = tf.stack([audio_contents, s_label])
        return tf.io.serialize_tensor(stacked_tensor)

    def deserialize_from_tfrecords(stacked_tensor):
        # audio_contents is already a string
        t = tf.io.parse_tensor(stacked_tensor, tf.string)
        audio_contents = t[0]
        label = tf.io.parse_tensor(t[1], tf.int32)
        return audio_contents, label


class AudioTarReader:
    def __init__(self, filename, sr=48000):
        assert os.path.isfile(filename), f'{filename} does not exist'
        self.audio_tarfile = filename
        # we know that sampling rate is 48kHz
        self.sr = sr

        if self.audio_tarfile.endswith('.tar'):
            self.audios_tar = tarfile.open(self.audio_tarfile, 'r')
        elif self.audio_tarfile.endswith('.tar.gz'):
            self.audios_tar = tarfile.open(self.audio_tarfile, "r:gz")
        else:
            raise ValueError('File must be .tar or .tar.gz')

        self.tar_file_list = [x for x in tqdm(self.audios_tar, desc=filename)]

        # get train.tsv, dev.tsv, test.tsv info
        self._retrieve_info()

    def retrieve_per_user_data(self, split='train'):
        """ Reads train/dev/test audio data from file

        Arguments:

        split - split to read (train, dev, test)
        """
        valid_splits = ['train', 'dev', 'test']
        assert split in valid_splits, \
            f'Invalid split: {split}. Must be one of {valid_splits}'

        split = split + '.tsv'

        # learn to map files to id's
        path_to_client_dict = dict(zip(
            self.data_files[split].path,
            self.data_files[split].client_id
        ))
        audio_content = {}
        for x in tqdm(self.tar_file_list):
            name_split = x.name.split('/')
            cur_id = path_to_client_dict.get(name_split[-1], False)
            if cur_id:
                audio_data = self.audios_tar.extractfile(x).read()
                cur_id_dict = audio_content.get(cur_id, [])
                cur_id_dict.append(audio_data)
                audio_content[cur_id] = cur_id_dict
        return audio_content

    # aux functions
    def _retrieve_info(self):
        """ Retrieves train/dev/test information from file

        Arguments:

        audios_tar - zip file with audios
        tar_file_list - list of files in zip file
        """
        self.data_files = {
            'train.tsv': None,
            'dev.tsv': None,
            'test.tsv': None,
            # 'validated.tsv': None
        }
        n_files = len(self.data_files.keys())
        cur_files = 0

        for x in self.tar_file_list:
            for k in self.data_files:
                if x.name.endswith(k):
                    with self.audios_tar.extractfile(x) as f:
                        df = pd.read_csv(f, sep='\t')
                        self.data_files[k] = df
                    cur_files += 1
            if cur_files == n_files:
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder", type=str,
        help="Folder containing Mozilla Speech files to be parsed"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # sampling rate for Mozilla dataset is 48 kHz
    sr = 48000
    assert os.path.isdir(args.folder), \
        f'Folder does not exit: {args.folder}'

    print(f'Searching folder: {args.folder}')

    # prep information
    files = [
        x for x in os.listdir(args.folder)
        if x.endswith('.tar') or x.endswith('.tar.gz')
    ]
    print(f'Generating tfrecords for: {files}')
    files = [os.path.join(args.folder, x) for x in files]
    for file in files:
        atr = None
        for split in ['train', 'dev', 'test']:
            target_name = file + f'_{split}.tfrecords.gzip'
            if not os.path.isfile(target_name):
                # only read on demand
                if atr is None:
                    atr = AudioTarReader(file)

                audio_content = atr.retrieve_per_user_data(split=split)

                # temp code
                """
                # the tfrecords from English is too big for Colab
                # this is a partial solution
                print(f'********** {len(audio_content)}')
                keys = list(audio_content.keys())
                keys = keys[0:len(keys) // 2]
                audio_content2 = {}
                for k in keys:
                    audio_content2[k] = audio_content[k]
                audio_content = audio_content2
                print(f'********** {len(audio_content)}')
                """

                pia = PersonIdAudio(audio_content, sr, verbose=1)
                audio_dataset = pia.get_tf_dataset()
                tfrecords_file = pia.save_tfrecords_file(
                    target_name
                )
                print(f'Saved {tfrecords_file}')
            else:
                print(f'Skipping {target_name}')


if __name__ == "__main__":
    # execute only if run as a script
    main()
