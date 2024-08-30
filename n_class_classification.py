import os.path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# configs
import configs.n_class.n_class_config_libri_libri as libri_libri_config
import configs.n_class.n_class_config_libri_tsp as libri_tsp_config
import configs.n_class.n_class_config_tsp_libri as tsp_libri_config
import configs.n_class.n_class_config_tsp_tsp as tsp_tsp_config
from data_loading import construct_dataset_n_class_models, shuffle_dataset


def train_classifier(d_config, classifier_type):
    # load train dataset
    train_X, train_Y = construct_dataset_n_class_models(basedir=d_config.base_path, paths_real=d_config.train_paths_orig,
                                                        paths_ai=d_config.train_paths_model,
                                                        formats=d_config.orig_data_formats_train, sr=d_config.sr,
                                                        highpass=d_config.highpass, log_scale=d_config.log_scale)

    train_X, train_Y = shuffle_dataset(train_X, train_Y, seed=d_config.seed)

    scaler = StandardScaler().fit(train_X)  # Normalize training data by mean + variance
    train_X = scaler.transform(train_X)

    if nonlinear_transf == "Nystroem":
        feature_map_nystroem = Nystroem()
        train_X = feature_map_nystroem.fit_transform(train_X)
        print("INFO -- applying Nystroem transf.")

    if classifier_type == "log_reg":
        print("INFO -- Construct LogisticRegression")
        classifier = LogisticRegression(random_state=d_config.seed, max_iter=MAX_ITER, multi_class=MULTI_CLASS,
                                        solver=SOLVER).fit(
            train_X, train_Y)

    elif classifier_type == "lin_svm":
        print("INFO -- Construct LinearSVC")
        classifier = LinearSVC(max_iter=MAX_ITER, multi_class=MULTI_CLASS).fit(train_X, train_Y)  # uses liblinear

    else:
        print("Error -- no valid classifier type was specified. Got {}. Exiting".format(classifier_type))
        exit(-1)
    return classifier


def test_classifier(d_config, classifier):
    test_X, test_Y = construct_dataset_n_class_models(basedir=d_config.base_path, paths_real=d_config.test_paths_orig,
                                                      paths_ai=d_config.test_paths_model,
                                                      formats=d_config.orig_data_formats_test, sr=d_config.sr,
                                                      highpass=d_config.highpass, log_scale=d_config.log_scale,
                                                      snr=d_config.noise_snr_test, compr=d_config.compr_test,
                                                      test_sr=d_config.resampling_sr_test)

    scaler = StandardScaler().fit(test_X)  # Normalize training data by mean + variance
    test_X = scaler.transform(test_X)

    if nonlinear_transf == "Nystroem":
        feature_map_nystroem = Nystroem()
        test_X = feature_map_nystroem.fit_transform(test_X)
        print("INFO -- applying Nystroem transf.")

    score = classifier.score(test_X, test_Y)
    predictions = classifier.predict(test_X)

    return score, test_Y, predictions


if __name__ == "__main__":
    CONFIGS = [libri_libri_config, libri_tsp_config, tsp_libri_config, tsp_tsp_config]
    OUTPUT_DIR = "classifier_results_n_class/"

    COLUMNS_CSV = ['model', 'train_set_orig', 'train_set_ai', 'test_set_orig', 'test_set_ai', 'accuracy',
                   'noise_post_proc_snr_test', 'compr_post_proc_test', 'resampling_post_proc_test', 'highpass',
                   'log_scale', 'solver', 'max_iter', 'mode']
    LABELS_TABLE = ['Real', 'Lyra_9.2', 'Encodec_24', 'IRVQGAN']

    SEEDS = [377, 1108, 1711, 2214, 2890]  # randomly chosen

    SOLVER = 'liblinear'
    MAX_ITER = 500  # 1000
    MULTI_CLASS = 'ovr'
    nonlinear_transf = "none"
    classifier_type = "log_reg"  # "lin_svm"

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

        OUT_CSV = os.path.join(OUTPUT_DIR, f"log_regression_results_n_class_{s}.csv")
        print("INFO -- Running Experiments for SEED: {}".format(s))

        # do for every config
        for config in CONFIGS:
            print("INFO -- Current Config: {}".format(config))
            print("INFO -- Overwriting config seed with: {}".format(s))
            config.seed = s
            classifier = train_classifier(config, classifier_type=classifier_type)

            score, test_Y, predictions = test_classifier(config, classifier)
            cm = metrics.confusion_matrix(test_Y, predictions)
            report = classification_report(test_Y, predictions,
                                           target_names=LABELS_TABLE,
                                           digits=8, output_dict=True)
            print(report)
            print(cm)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{config.data_conf}_classification_report_samplerate_{config.sr}_s_{s}.csv"))
            cm_df = pd.DataFrame(cm, index=LABELS_TABLE, columns=LABELS_TABLE)
            cm_df.to_csv(os.path.join(OUTPUT_DIR, f"{config.data_conf}_confusion_mat_samplerate_{config.sr}_s_{s}.csv"))
            print('Test score:', "{:.4f}".format(score))

            model_names.append(config.model_name)
            train_set_paths_orig.append(config.train_paths_orig)
            train_set_paths_ai.append(config.train_paths_model)
            test_set_paths_orig.append(config.test_paths_orig)
            test_set_paths_ai.append(config.test_paths_model)
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
                       'mode': MULTI_CLASS}

        df = pd.DataFrame(result_dict, columns=COLUMNS_CSV)
        df.to_csv(OUT_CSV, index=False, mode='a', header=not os.path.exists(OUT_CSV))
        print(f'DataFrame has been written to {OUT_CSV}')
