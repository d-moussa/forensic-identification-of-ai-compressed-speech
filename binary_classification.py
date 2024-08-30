import os.path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from data_loading import load_datasets, shuffle_dataset


# libri libri
import configs.binary.libri_libri.lyra as libri_libri_lyra
import configs.binary.libri_libri.encodec as libri_libri_encodec
import configs.binary.libri_libri.irvqgan as libri_libri_irvqgan


# libri tsp
import configs.binary.libri_tsp.irvqgan as libri_tsp_encodec
import configs.binary.libri_tsp.irvqgan as libri_tsp_irvqgan
import configs.binary.libri_tsp.lyra as libri_tsp_lyra

# tsp libri
import configs.binary.tsp_libri.encodec as tsp_libri_encodec
import configs.binary.tsp_libri.irvqgan as tsp_libri_irvqgan
import configs.binary.tsp_libri.lyra as tsp_libri_lyra

# tsp tsp
import configs.binary.tsp_tsp.encodec as tsp_tsp_encodec
import configs.binary.tsp_tsp.irvqgan as tsp_tsp_irvqgan
import configs.binary.tsp_tsp.lyra as tsp_tsp_lyra


if __name__ == "__main__":
    CONFIGS = (
                [tsp_tsp_encodec, tsp_tsp_irvqgan, tsp_tsp_lyra ] +
                [tsp_libri_encodec, tsp_libri_irvqgan, tsp_libri_lyra ] +
                [libri_tsp_encodec, libri_tsp_irvqgan, libri_tsp_lyra] +
                [libri_libri_encodec, libri_libri_irvqgan, libri_libri_lyra ]
             )
    OUTPUT_DIR = "classifier_results_binary/"
    COLUMNS_CSV = ['model', 'train_set_orig', 'train_set_ai', 'test_set_orig', 'test_set_ai', 'accuracy', 'noise_post_proc_snr_test',
                   'compr_post_proc_test', 'resampling_post_proc_test',  'highpass', 'log_scale', 'solver', 'max_iter', 'mode']
    SOLVER = 'liblinear'
    MAX_ITER = 500 #1000
    SEEDS = [377, 1711, 2890, 2214, 1108]  # randomly chosen
    nonlinear_transf = "none"
    classifier_type = "log_reg" #"lin_svm"

    model_names = []
    train_set_paths_orig = []
    train_set_paths_ai = []
    test_set_paths_orig = []
    test_set_paths_ai = []
    accuracies = []
    noise_params = []
    compr_params = []
    resampling_params = []
    highpass_flags = []
    logscale_flags = []

    # do for every seed
    print("INFO -- Writing to: {}".format(OUTPUT_DIR))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for s in SEEDS:
        np.random.seed(s)
        OUT_CSV = os.path.join(OUTPUT_DIR, f"log_regression_results_binary_model_sr_{s}.csv")
        print("INFO -- Running Experiments for SEED: {}".format(s))

        # do for all train/test combinations
        for config in CONFIGS:
            print("INFO -- Current Config: {}".format(config))
            print("INFO -- Overwriting config seed with: {}".format(s))
            config.seed = s

            # train classifier
            for (train_data_real, train_data_ai), format_train in zip(zip(config.train_paths_orig, config.train_paths_model), config.orig_data_formats_train):

                train_X, train_Y = load_datasets(path_real=os.path.join(config.base_path,train_data_real),
                                                 path_ai=os.path.join(config.base_path, train_data_ai),
                                                 format=format_train,
                                                 sr=config.model_sr,
                                                 highpass=config.highpass,
                                                 log_scale=config.log_scale)

                train_X, train_Y = shuffle_dataset(train_X, train_Y, seed=config.seed)

                scaler = StandardScaler().fit(train_X) # Normalize training data by mean + variance
                train_X = scaler.transform(train_X)

                if nonlinear_transf == "Nystroem":
                    feature_map_nystroem = Nystroem()
                    train_X = feature_map_nystroem.fit_transform(train_X)
                    print("INFO -- applying Nystroem transf.")

                if classifier_type == "log_reg":
                    print("INFO -- Construct LogisticRegression")
                    classifier = LogisticRegression(random_state=config.seed, max_iter=MAX_ITER, solver=SOLVER).fit(train_X, train_Y)
                elif classifier_type=="lin_svm":
                    print("INFO -- Construct LinearSVC")
                    classifier = LinearSVC(max_iter=MAX_ITER).fit(train_X, train_Y)
                else:
                    print("Error -- no valid classifier type was specified. Got {}. Exiting".format(classifier_type))
                    exit(-1)

                # test classifier
                for (test_data_real, test_data_ai), format_test in zip(zip(config.test_paths_orig, config.test_paths_model), config.orig_data_formats_test):

                    test_X, test_Y = load_datasets(path_real=os.path.join(config.base_path, test_data_real),
                                                 path_ai=os.path.join(config.base_path, test_data_ai),
                                                 format=format_test,
                                                 sr=config.model_sr,
                                                 highpass=config.highpass,
                                                 log_scale=config.log_scale,
                                                 snr=config.noise_snr_test,
                                                 compr=config.compr_test,
                                                 test_sr=config.resampling_sr_test)

                    scaler = StandardScaler().fit(test_X)  # Normalize training data by mean + variance
                    test_X = scaler.transform(test_X)

                    if nonlinear_transf == "Nystroem":
                        feature_map_nystroem = Nystroem()
                        test_X = feature_map_nystroem.fit_transform(test_X)
                        print("INFO -- applying Nystroem transf.")

                    score = classifier.score(test_X, test_Y)
                    print('Test score:', "{:.4f}".format(score))

                    model_names.append(config.model_name)
                    train_set_paths_orig.append(train_data_real)
                    train_set_paths_ai.append(train_data_ai)
                    test_set_paths_orig.append(test_data_real)
                    test_set_paths_ai.append(test_data_ai)
                    accuracies.append("{:.8f}".format(score))
                    noise_params.append(config.noise_snr_test)
                    compr_params.append(config.compr_test)
                    resampling_params.append(config.resampling_sr_test)
                    highpass_flags.append(config.highpass)
                    logscale_flags.append(config.log_scale)

        result_dict = {'model': model_names,
                       'train_set_orig': train_set_paths_orig,
                       'train_set_ai': train_set_paths_ai,
                       'test_set_orig': test_set_paths_orig,
                       'test_set_ai': test_set_paths_ai,
                       'accuracy': accuracies,
                       'noise_post_proc_snr_test': noise_params,
                       'compr_post_proc_test': compr_params,
                       'resampling_post_proc_test': resampling_params,
                       'highpass': highpass_flags,
                       'log_scale': logscale_flags,
                       'solver': SOLVER,
                       'max_iter': MAX_ITER,
                       'mode': 'binary'}

        df = pd.DataFrame(result_dict, columns=COLUMNS_CSV)
        df.to_csv(OUT_CSV, index=False, mode='w', header=not os.path.exists(OUT_CSV))
        print(f'DataFrame has been written to {OUT_CSV}')








