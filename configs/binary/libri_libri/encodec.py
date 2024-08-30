model_name = "encodec_24kHz" # model identifier (for results file)
base_path = "data/" # path to directory containing all audio data subdirectories

# training data paths (relative to base_path)
train_paths_model = ["ai_compressed_data/librispeech/encodec/bw_{}/".format(x) for x in [1.5, 3, 6, 12, 24]]  # neural codec training data
train_paths_orig = [*["uncompressed_data/librispeech_partial_2s_24kHz/"] * len(train_paths_model)]  # uncompressed version of training data

# test data paths (relative to base_path)
data_set = "librispeech_test"
test_paths_model = (  # list of directories containing the audio samples to test on
                    [f"ai_compressed_data/{data_set}/lyra/bw_{x}/" for x in [3200, 6000, 9200]] +
                    [f"ai_compressed_data/{data_set}/encodec/bw_{x}/" for x in [1.5, 3, 6, 12, 24]] +
                    [f"ai_compressed_data/{data_set}/irvqgan/bw_8/"]
                   )
test_paths_orig = [*["uncompressed_data/librispeech_partial_test_2s_24kHz/"] * len(test_paths_model)]  # uncompressed version of test data

# audio format specifications
orig_data_formats_train = [*["flac"] * len(train_paths_orig)]
orig_data_formats_test = [*["flac"] * len(test_paths_model)]

# sampling rate of neural codec
model_sr = 24000

# FFT feature options (set both to True to reproduce paper results)
highpass = True
log_scale = True

# random seed
seed = 42

# degradations (degradations are evaluated one at a time in the paper. Adjust the order of degradation application in
# data_loading.prepair_data_fft to your requirements to test several degradations at once)
noise_snr_test = None
compr_test = (None, None)  # <(format, bitrate)>
resampling_sr_test = None
