import os
# suppress tensorflow messages
# include this if convenient
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import train_speech_id_model


def main():
    threshold = 0.83
    audio_files = [x for x in os.listdir() if x.endswith('mp3')]
    audio_embeddings = []
    print(f'Comparing {audio_files}')
    model = tf.keras.models.load_model('speech-id-model-110')
    target_rate = 48000

    for file in audio_files:
        cur_data = tfio.audio.AudioIOTensor(file)
        print(f'Processing {file} with sample rate of {cur_data.rate}')
        audio_data = cur_data.to_tensor()[:, 0]
        if cur_data.rate != target_rate:
            print(f'Sampling rate is not {target_rate}. Resampling...')
            audio_data = tfio.audio.resample(
                audio_data,
                tf.cast(cur_data.rate, tf.int64),
                tf.cast(target_rate, tf.int64),
            )
        # set batch size to 1, extract first element
        cur_emb = model.predict(
            tf.expand_dims(audio_data, axis=0)
        )[0]
        audio_embeddings.append(cur_emb)

    for p in range(len(audio_files)):
        for q in range(p + 1, len(audio_files)):
            f1 = audio_files[p]
            f2 = audio_files[q]
            distance = np.linalg.norm(
                audio_embeddings[p] - audio_embeddings[q]
            )
            if distance < threshold:
                conclusion = 'Same person:'
            else:
                conclusion = 'Different people:'
            print(f'{f1} and {f2}: {conclusion} {distance}')


if __name__ == "__main__":
    # execute only if run as a script
    main()
