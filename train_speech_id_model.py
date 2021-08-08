import os

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras import layers as L

from create_audio_tfrecords import AudioTarReader, PersonIdAudio


# make compatible with tensorflow 2.4
# this was supposed to be tfio.audio.spectrogram
def spectrogram_fn(input, nfft, window, stride, name=None):
    """
    Create spectrogram from audio.
    Args:
      input: An 1-D audio signal Tensor.
      nfft: Size of FFT.
      window: Size of window.
      stride: Size of hops between windows.
      name: A name for the operation (optional).
    Returns:
      A tensor of spectrogram.
    """

    # TODO: Support audio with channel > 1.
    return tf.math.abs(
        tf.signal.stft(
            input,
            frame_length=window,
            frame_step=stride,
            fft_length=nfft,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
    )


def normalized_mel_spectrogram(x, sr=48000, n_mel_bins=80):
    spec_stride = 256
    spec_len = 1024

    spectrogram = spectrogram_fn(
        x, nfft=spec_len, window=spec_len, stride=spec_stride
    )

    num_spectrogram_bins = spec_len // 2 + 1  # spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz = 80.0, 10000.0
    num_mel_bins = n_mel_bins
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    avg = tf.math.reduce_mean(log_mel_spectrograms)
    std = tf.math.reduce_std(log_mel_spectrograms)

    return (log_mel_spectrograms - avg) / std


def BaseSpeechEmbeddingModel(inputLength=None, rnn_func=L.LSTM, rnn_units=128):
    # input is the first channel of the decoded mp3, ie,
    # tfio.audio.decode_mp3(data)[:, 0]

    inp = L.Input((inputLength,), name='input')
    mel_spec = L.Lambda(
        lambda z: normalized_mel_spectrogram(z), name='normalized_spectrogram'
    )(inp)

    # receive normalized mel spectrogram as input instead
    # inp = L.Input((inputLength, n_mel_bins), name='input')
    # mel_spec = inp

    # normalize the spectrogram
    # mel_spec = L.BatchNormalization()(mel_spec)
    # mel_spec = L.LayerNormalization()(mel_spec)

    x = L.Bidirectional(
        rnn_func(rnn_units,
                 return_sequences=True)
    )(mel_spec)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(
        rnn_func(rnn_units, return_sequences=False)
    )(x)  # [b_s, seq_len, vec_dim]

    # No activation on final dense layer
    x = L.Dense(rnn_units, activation=None)(x)
    # L2 normalize embeddings
    # note: L2 returns normalized, norm
    x = L.Lambda(lambda z: tf.math.l2_normalize(z, axis=1), name='output')(x)

    output = x

    model = Model(inputs=[inp], outputs=[output])
    return model


def main():
    # this is useful to know how far in
    # the batch_size we can go
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_files = [x for x in os.listdir('data')
                   if x.endswith('train.tfrecords.gzip')]
    train_files = [os.path.join('data', x) for x in train_files]

    # pick one file
    sorted(train_files)
    train_file = train_files[0]
    print(f'Training with {train_file}')

    # check if tfrecords file is OK
    # notice GZIP compression + the deserialization function map
    tfrecords_audio_dataset = tf.data.TFRecordDataset(
        train_file, compression_type='GZIP',
        # num_parallel_reads=4
    ).map(PersonIdAudio.deserialize_from_tfrecords)

    # count number of records
    n_train_samples = sum(1 for _ in tfrecords_audio_dataset)
    print(n_train_samples)

    n_mel_bins = 80

    m = BaseSpeechEmbeddingModel()
    m.summary()

    # 9 Gb  GPU RAM with 256
    batch_size = 128 * 3

    return_mel_spec = False

    def mp3_decode_fn(audio_bytes, audio_class):
        # check if limiting output size helps
        # return tfio.audio.decode_mp3(audio_bytes)[:, 0], audio_class
        # audio_data = tfio.audio.decode_mp3(audio_bytes)[:, 0]
        audio_data = tfio.audio.decode_mp3(audio_bytes)[0:int(48000 * 5), 0]
        if return_mel_spec:
            audio_data = normalized_mel_spectrogram(audio_data)
        return audio_data, audio_class

    train_set = tfrecords_audio_dataset.map(
            # Reduce memory usage
            mp3_decode_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(
            10 * batch_size,
            reshuffle_each_iteration=True
        ).repeat(
        ).padded_batch(  # Vectorize your mapped function
            batch_size,  # batch size
            # padded_shapes=([None, None], []),
            padded_shapes=([None], []),
            drop_remainder=True
        ).prefetch(  # Overlap producer and consumer works
            tf.data.AUTOTUNE
        )

    m.compile(
        optimizer=tf.keras.optimizers.Adam(0.0006),
        loss=tfa.losses.TripletSemiHardLoss()
    )

    # m = tf.keras.models.load_model('speech_id_model')

    os.makedirs('temp', exist_ok=True)
    checkpoint_filepath = 'temp/cp-{epoch:04d}.ckpt'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    history = m.fit(
        train_set,
        steps_per_epoch=n_train_samples // batch_size,
        epochs=200,
        callbacks=[model_checkpoint_callback]
    )

    m.save('speech_id_model')
    m.save('speech_id_model.h5')


if __name__ == "__main__":
    # execute only if run as a script
    main()
