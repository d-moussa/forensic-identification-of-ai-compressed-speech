model_name = "irvqgan_24kHz"

base_path = "data/"
train_paths_model = ["ai_compressed_data/tsp/irvqgan/bw_8/" ]
train_paths_orig = ["uncompressed_data/tsp_partial_2s_24kHz/"]


data_set = "tsp_test"
test_paths_model = ([f"ai_compressed_data/{data_set}/lyra/bw_{x}/" for x in [3200, 6000, 9200]] +
                    [f"ai_compressed_data/{data_set}/encodec/bw_{x}/" for x in [1.5, 3, 6, 12, 24]] +
                    [f"ai_compressed_data/{data_set}/irvqgan/bw_8/"])
test_paths_orig = [*["uncompressed_data/tsp_partial_test_2s_24kHz/"]*len(test_paths_model)]


orig_data_formats_train = [*["wav"]*len(train_paths_orig)]
orig_data_formats_test = [*["wav"]*len(test_paths_model)]
model_sr = 24000


highpass = True
log_scale = True

seed = 42


noise_snr_test = None
compr_test = (None, None)   # <(format, bitrate)>
resampling_sr_test = None
