import os
import random
from random import shuffle

import math
import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
import torchaudio.functional as F


def apply_noise(wave_list, snr_db):
    for i in range(len(wave_list)):
        if snr_db is None:
            continue

        noise = (0.001 ** 0.5) * torch.randn(1, wave_list[i].shape[1])
        sig_power = wave_list[i].norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / sig_power
        wave_list[i] = (scale * wave_list[i] + noise) / 2

        augmented_power = wave_list[i].norm(p=2)
        a_scale = sig_power / augmented_power
        wave_list[i] = wave_list[i] * a_scale

    return wave_list


def apply_codec(wave_list, sr, bitrate, format="mp3", processes=1):
    def worker(w, wav_start_index, queue, sr):
        for wav_subset_index in range(len(w)):
            if format == "mp3":
                w[wav_subset_index] = F.apply_codec(w[wav_subset_index], sample_rate=sr, format=format,
                                                    compression=bitrate)[:, 832:]
            elif format == "vorbis":
                w[wav_subset_index] = F.apply_codec(w[wav_subset_index], sample_rate=sr, format=format,
                                                    compression=bitrate)
        queue.put((w, wav_start_index))
        done.wait()

    i = 0
    q = mp.Queue()
    proc_list = []
    rets = []
    wave_len = len(wave_list)
    per_proc_size = int(np.ceil(wave_len / processes))

    for _ in range(processes):
        p_wave_list = wave_list[i:per_proc_size + i]
        done = mp.Event()
        p = mp.Process(target=worker, args=(p_wave_list, i, q, sr))
        proc_list.append([p, done])
        p.start()
        i += per_proc_size

    for _ in range(processes):
        ret = q.get()
        rets.append(ret)
        del ret

    rets.sort(key=lambda x: x[1])
    retVal = []
    for r in rets:
        retVal += r[0]

    for p, done in proc_list:
        done.set()
        p.join()
        p.close()
    wave_list = retVal
    return wave_list


def log_scale(array, epsilon=1e-12):
    array = torch.abs(array)
    array += epsilon
    array = torch.log(array)
    return array


def highpass(audio):
    assert len(audio.shape) == 2
    audio = audio.squeeze()
    kernel = np.array([-1.0, 2.0, -1.0])
    audio_filtered = np.convolve(audio, kernel, mode='same')
    return torch.tensor(audio_filtered.reshape(1, -1))


def construct_dataset_n_class_models(basedir, paths_real, paths_ai, formats, sr, highpass, log_scale, snr=None,
                                     compr=(None, None), use_float_32=False, stft=False, test_sr=None):
    orig_Xs = []
    comp_Xs = []

    # load uncompressed sets and merge to one tensor
    for path_real, format in zip(paths_real, formats):
        orig_Xs.append(prepair_data_fft(os.path.join(basedir, path_real), in_file_format=format, target_sr=sr,
                                        to_highpass=highpass, to_log=log_scale, snr=snr, compr=compr,
                                        use_float32=use_float_32, stft=stft,
                                        test_sr=test_sr))
    if len(orig_Xs) > 1:
        orig_X = torch.vstack(orig_Xs)
    else:
        orig_X = orig_Xs[0]

    # load all ai_compr sets
    for path_ai in paths_ai:
        comp_Xs.append(
            prepair_data_fft(os.path.join(basedir, path_ai), in_file_format="wav", target_sr=sr, to_highpass=highpass,
                             to_log=log_scale, snr=snr, compr=compr, use_float32=use_float_32, stft=stft,
                             test_sr=test_sr))

    # generate labels (real, ai_codec1-n)
    orig_Y = torch.zeros(orig_X.shape[0])
    if use_float_32:
        orig_Y = orig_Y.to(torch.float32)
    comp_Ys = []
    for i, data in enumerate(comp_Xs):
        c_y = torch.full((data.shape[0],), i + 1)
        if use_float_32:
            c_y = c_y.to(torch.float32)
        comp_Ys.append(c_y)
    Y = torch.hstack([orig_Y] + comp_Ys)

    # stack all ai_compr sets
    comp_X = torch.vstack(comp_Xs)
    X = torch.vstack((orig_X, comp_X))

    return X, Y


def load_directory(dir_path, in_file_format, target_sr):
    wavs = []
    for fn in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, fn)): continue
        x, sr_orig = torchaudio.load(os.path.join(dir_path, fn), format=in_file_format)
        if target_sr is not None and target_sr != sr_orig:
            x = torchaudio.functional.resample(x, new_freq=target_sr, orig_freq=sr_orig)
        wavs.append(x)
    return wavs


def load_and_resample(dir_path, in_file_format, test_sr, target_sr):
    wavs = []
    for fn in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, fn)): continue
        x, sr_orig = torchaudio.load(os.path.join(dir_path, fn), format=in_file_format)
        if test_sr is not None:
            x = torchaudio.functional.resample(x, new_freq=test_sr, orig_freq=sr_orig)
            x = torchaudio.functional.resample(x, new_freq=target_sr, orig_freq=test_sr)
        wavs.append(x)
    return wavs


def prepair_data_fft(in_dir, in_file_format, target_sr, to_highpass=True, to_log=True, snr=None, compr=(None, None),
                     use_float32=False, stft=False, test_sr=None):
    if test_sr is not None:
        print("INFO -- Apply Resampling with SR: {}".format(test_sr))
        loaded_list_tensor = load_and_resample(in_dir, in_file_format, test_sr, target_sr)
    else:
        loaded_list_tensor = load_directory(in_dir, in_file_format, target_sr)

    if snr is not None:
        print("INFO -- Apply Whitenoise with SNR: {}".format(snr))
        loaded_list_tensor = apply_noise(loaded_list_tensor, snr)

    format, bitrate = compr
    if format is not None:
        print("INFO -- Apply {} Compression with Bitrate: {}".format(format, bitrate))
        loaded_list_tensor = apply_codec(loaded_list_tensor, sr=target_sr, bitrate=bitrate, format=format, processes=8)
    X = torch.stack(loaded_list_tensor)

    if to_highpass:
        X = torch.stack([highpass(x) for x in X])
    if stft:
        transform = torchaudio.transforms.Spectrogram(n_fft=800)
        X = torch.stack([transform(wav[0, :]) for wav in X])
    else:
        X = torch.stack([torch.fft.rfft(wav[0, :]).abs() for wav in X])

    if to_log:
        X = log_scale(X)
    if use_float32:
        X = X.to(torch.float32)
    return X


def load_datasets(path_real, path_ai, format, sr, highpass, log_scale, snr=None, compr=(None, None), test_sr=None):
    orig_X = prepair_data_fft(path_real, in_file_format=format, target_sr=sr, to_highpass=highpass, to_log=log_scale,
                              snr=snr, compr=compr, test_sr=test_sr)
    comp_X = prepair_data_fft(path_ai, in_file_format="wav", target_sr=sr, to_highpass=highpass, to_log=log_scale,
                              snr=snr, compr=compr, test_sr=test_sr)
    X = torch.vstack((comp_X, orig_X))
    Y = torch.hstack(
        (torch.ones(len(comp_X)), torch.zeros(len(orig_X))))  # labels (ai_compr: 1, orig: 0)
    return X, Y


def shuffle_dataset(X, Y, seed=None):
    if seed is not None:
        random.seed(seed)
    idx_list = list(range(Y.shape[0]))
    shuffle(idx_list)
    X = X[idx_list, :]
    Y = Y[idx_list]
    return X, Y
