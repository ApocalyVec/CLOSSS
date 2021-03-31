import os

import matplotlib
from sklearn.manifold import TSNE

from helpers import listdir_fullpath, get_embeddings_from_wav

matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys

import sklearn
from lpctorch import LPCCoefficients
from sklearn.decomposition import PCA

dataset_root = '/home/apocalyvec/Data/Speech/LibriSpeech ASR corpus'  # change this to your data root
num_sample_per_spk = 15
num_speaker = 10

if __name__ == '__main__':
    # put in random seed
    np.random.seed(42)
    # set plotting parameters
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    print_args(args, parser)
    if not args.no_sound:  # check if there is audio output device attached to the system
        import sounddevice as sd

    # load in the model checkpoints ###################################################################################
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

    # get 10 samples from each speaker  ###############################################################################
    # load in samples from dev set
    data_root = os.path.join(dataset_root, 'dev-clean', 'LibriSpeech', 'dev-clean')
    speakers = list(np.random.choice(os.listdir(data_root), size=num_speaker))
    speaker_directories = [os.path.join(data_root, spk) for spk in speakers]

    spk_sample_dir_pair = [(spk, listdir_fullpath(spk_dir)[0]) for spk, spk_dir in zip(speakers, speaker_directories)]
    spk_sample_files_pair = [(spk, listdir_fullpath(smpl_dir)) for spk, smpl_dir in spk_sample_dir_pair]
    # filter out txt files
    spk_sample_files_pair = [(spk, [x for x in smpl_files if '.flac' in x]) for spk, smpl_files in spk_sample_files_pair]
    # randomly choose a few
    spk_sample_files_pair = [(spk, np.random.choice(smpl_files, size=num_sample_per_spk))  for spk, smpl_files in spk_sample_files_pair]

    # encode the samples
    print('encoding samples, this might take a while...')
    spk_sample_embed_pair = [(spk, [get_embeddings_from_wav(x, encoder) for x in smpl_files]) for spk, smpl_files in spk_sample_files_pair]
    # process generated speech #######################################################################################
    # sample sentence to be synthesized
    text = 'The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents.'
    # inference LPC spectrogram based on input text and speaker embeddings
    specs = synthesizer.synthesize_spectrograms([text] * num_sample_per_spk * num_speaker, [item for _, embeds in spk_sample_embed_pair for item in embeds])
    # vocode from the inferred LPC
    print('Vocoding, this might take a while...')
    generated_wavs = [vocoder.infer_waveform(spec) for spec in specs]
    # pad the synthesized wav so be the same length as
    generated_wavs_padded = [np.pad(g_wav, (0, synthesizer.sample_rate), mode="constant") for g_wav in generated_wavs]
    # evaluate the speaker embeddings of the generated wav
    print('encoding generated samples, this might take a while...')
    embeds_generated = [encoder.embed_utterance(g_wav_padded) for g_wav_padded in generated_wavs_padded]

    # Perform PCA #######################################################################################

    pca = PCA(n_components=2)
    # need to flatten list for PCA
    principal_components_original = pca.fit_transform(np.array([item for _, embeds in spk_sample_embed_pair for item in embeds]))
    principal_components_generated = pca.transform(np.array(embeds_generated))
    # group for every speaker
    principal_components_original_grouped = list(zip(*[iter(principal_components_original)]*num_sample_per_spk))
    principal_components_generated_grouped = list(zip(*[iter(principal_components_generated)]*num_sample_per_spk))

    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('rainbow')  # rotating color wheel
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    for spk, sample_pca, generated_sample_pca, clr in zip(speakers, principal_components_original_grouped, principal_components_generated_grouped, colors):
        a1 = np.array(sample_pca)
        a2 = np.array(generated_sample_pca)
        plt.scatter(a1[:, 0], a1[:, 1], color=clr, marker='o', label='Speaker ' + spk)
        plt.scatter(a2[:, 0], a2[:, 1], color=clr,  marker='x')
    plt.title('PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Perform T_SNE #######################################################################################
    # for p in range(5, 55, 5):
    #     print(p)
    p = 25
    tsne = TSNE(n_components=2, verbose=1, perplexity=p, n_iter=600)
    tsne_results = tsne.fit_transform(np.concatenate([np.array([item for _, embeds in spk_sample_embed_pair for item in embeds]), np.array(embeds_generated)]))

    tsne_results_original = tsne_results[:num_sample_per_spk * num_speaker]
    tsne_results_generated = tsne_results[num_sample_per_spk * num_speaker:]
    # group for every speaker
    tsne_results_original_grouped = list(zip(*[iter(tsne_results_original)]*num_sample_per_spk))
    tsne_results_generated_grouped = list(zip(*[iter(tsne_results_generated)]*num_sample_per_spk))

    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('rainbow')  # rotating color wheel
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    for spk, sample_pca, generated_sample_pca, clr in zip(speakers, tsne_results_original_grouped, tsne_results_generated_grouped, colors):
        a1 = np.array(sample_pca)
        a2 = np.array(generated_sample_pca)
        plt.scatter(a1[:, 0], a1[:, 1], color=clr, marker='o', label='Speaker ' + spk)
        plt.scatter(a2[:, 0], a2[:, 1], color=clr,  marker='x')
    plt.title('t_SNE Perplexity=' + str(p))
    plt.xlabel('t-SNE 2D 1')
    plt.ylabel('t-SNE 2')
    plt.show()
