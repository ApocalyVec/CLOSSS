import matplotlib

matplotlib.use('module://backend_interagg')
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse

import sklearn
from lpctorch import LPCCoefficients
#
# reference_audio_path = 'speech_samples/n_sample.wav'  # change this to
# sentence_to_synthesize = 'The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-r", "--reference_audio_path", type=str,
                        default="speech_samples/ICouldNot.wav",
                        help="path to the speaker reference audio")
    parser.add_argument("-sts", "--sentence_to_synthesize", type=str,
                        default="The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents.",
                        help="The sentence to be synthesized")

    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    args = parser.parse_args()
    if not args.no_sound:  # check if there is audio output device attached to the system
        import sounddevice as sd

    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

    print('path is ' + args.reference_audio_path)
    # load in the sample audio
    original_wav, sampling_rate = librosa.load(args.reference_audio_path)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")

    ## Generating the spectrogram

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [args.sentence_to_synthesize]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print('Synthesizing ...')
    generated_wav = vocoder.infer_waveform(spec)

    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    fpath = 'synthesized_sample_' + args.reference_audio_path.split('/')[-1]
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32),
                             synthesizer.sample_rate)
    print('Saved as ' + fpath)
