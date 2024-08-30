data_conf = "TSP_Libri"
model_name = "encodec_24kHz-irvqgan_24kHz-lyra16kHz"

base_path = "data/"
train_paths_model = [
                    "ai_compressed_data/tsp/lyra/bw_{}/".format(9200),
                    "ai_compressed_data/tsp/encodec/bw_{}/".format(24),
                     "ai_compressed_data/tsp/irvqgan/bw_8/"
]
train_paths_orig = ["uncompressed_data/tsp_partial_2s_24kHz/"]

data_set = "librispeech_test"
test_paths_model = (
                    [f"ai_compressed_data/{data_set}/lyra/bw_{x}/" for x in [9200]]  +
                    [f"ai_compressed_data/{data_set}/encodec/bw_{x}/" for x in [24]] +
                    [ f"ai_compressed_data/{data_set}/irvqgan/bw_8/"]
                    )
test_paths_orig = [*["uncompressed_data/librispeech_partial_test_2s_24kHz/"]*1]

orig_data_formats_train = [*["wav"]*len(train_paths_model)]
orig_data_formats_test = [*["flac"]*len(test_paths_model)]
sr = 16000

highpass = True
log_scale = True

seed = 42

noise_snr_test = None
compr_test = (None, None)   # <(format, bitrate)>
resampling_sr_test = None