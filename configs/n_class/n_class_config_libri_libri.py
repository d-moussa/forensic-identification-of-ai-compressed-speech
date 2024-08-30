data_conf = "Libri_Libri"  # <TrainingSource_TestSource>, LibriSpeech (Libri) and TSP is used in the paper
model_name = "encodec_24kHz-irvqgan_24kHz-lyra16kHz"  # experiment specifier for results file
base_path = "data/"

# training data paths (relative to base_path) -> training set will be constructed from all sources
train_paths_model = [  # neural codec training data
                     "ai_compressed_data/librispeech/lyra/bw_{}/".format(9200),
                     "ai_compressed_data/librispeech/encodec/bw_{}/".format(24),
                     "ai_compressed_data/librispeech/irvqgan/bw_8/"
                    ]
train_paths_orig = ["uncompressed_data/librispeech_partial_2s_24kHz/"]  # uncompressed version of training data

# test data paths (relative to base_path)
data_set = "librispeech_test"
test_paths_model = (  # list of directories containing the audio samples to test on
                    [f"ai_compressed_data/{data_set}/lyra/bw_{x}/" for x in [9200]]  +
                    [f"ai_compressed_data/{data_set}/encodec/bw_{x}/" for x in [24]] +
                    [f"ai_compressed_data/{data_set}/irvqgan/bw_8/"]
                    )
test_paths_orig = [*["uncompressed_data/librispeech_partial_test_2s_24kHz/"]*1]  # uncompressed version of test data

# audio format specifications
orig_data_formats_train = [*["flac"]*len(train_paths_model)]
orig_data_formats_test = [*["flac"]*len(test_paths_model)]

# sampling rate for n-class task
sr = 16000

# FFT feature options (set both to True to reproduce paper results)
highpass = True
log_scale = True

# random seed
seed = 42

# degradations (degradations are evaluated one at a time in the paper. Adjust the order of degradation application in
# data_loading.prepair_data_fft to your requirements to test several degradations at once)
noise_snr_test = None
compr_test = (None, None)   # <(format, bitrate)>
resampling_sr_test = None