import os
import librosa


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_embeddings_from_wav(audio_path, encoder):
    # load in the sample audio
    original_wav, sampling_rate = librosa.load(audio_path)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    # encode the sample audio with the speaker embedding network
    embed = encoder.embed_utterance(preprocessed_wav)
    return embed