import argparse
import pandas as pd
import re

# Definitions for the Table column/rows
ALL_AI_MODEL_NAMES = ["Lyra_3.2", "Lyra_6", "Lyra_9.2", "Encodec_1.5", "Encodec_3", "Encodec_6",
                           "Encodec_12", "Encodec_24", "IRVQGAN"]
BINARY_SMALL_TABLE_NAMES = ["Lyra_6", "Encodec_1.5","Encodec_24", "IRVQGAN" ]

# Definitions for the Training-Test dataset combinations
TRAINING_DATA_SETS = ["Libri", "TSP"]
TEST_DATA_SETS = ["Libri", "TSP"]
TRAINING_TEST_SET_COMBINATIONS = ["Libri_Libri", "Libri_TSP", "TSP_Libri", "TSP_TSP"]

# Name mappings for naming conventions as expected in the CSV result files to process
MODEL_DIR_SPECIFIERS = {"lyra": "Lyra", "encodec": "Encodec", "irvqgan": "IRVQGAN"}
BW_SPECIFIERS = {"1.5": "1.5", "3":"3", "6":"6", "12":"12", "24":"24", "3200":"3.2", "6000":"6", "9200":"9.2"}

SET_SPECIFIERS = ["librispeech", "tsp"]

def create_empty_dataframe():
    df = pd.DataFrame(index=ALL_AI_MODEL_NAMES, columns=ALL_AI_MODEL_NAMES)
    return df
def get_model_specifier_for_table(cell_string):
    model_type = None
    model_bw = None
    # determine model type
    for name in MODEL_DIR_SPECIFIERS.keys():
        pattern = re.compile(re.escape('/'+name), re.IGNORECASE)
        match = re.search(pattern, cell_string)
        if match:
            model_type = MODEL_DIR_SPECIFIERS[name]
            break
    assert model_type is not None, (f"Could not find one of {MODEL_DIR_SPECIFIERS.keys()} in {cell_string}")
    # get bandwidth for model
    if model_type in ["Lyra", "Encodec"]:
        for bw in BW_SPECIFIERS.keys():
            pattern = re.compile(re.escape(bw+'/'))
            match = re.search(pattern, cell_string)
            if match:
                model_bw = BW_SPECIFIERS[bw]
                break
        assert model_bw is not None, (f"Could not find one of {BW_SPECIFIERS.keys()} in {cell_string}")
        return f"{model_type}_{model_bw}"
    else:
        return model_type

def get_dataset_combi_specifier_for_table(cell_string_train, cell_string_test, search_space):
    match_set1 = None
    match_set2 = None
    set_1_name = None
    set_2_name = None

    for name in search_space:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        match_set1 = re.search(pattern, cell_string_train)
        if match_set1:
            set_1_name = name
            break

    for name in search_space:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        match_set2 = re.search(pattern, cell_string_test)
        if match_set2:
            set_2_name = name
            break

    assert  match_set1 is not None and match_set2 is not None, (f"No valid dataset name was found in {search_space} for "
                                                                f"{cell_string_train} or {cell_string_test}")
    data_combi = f"{set_1_name}_{set_2_name}"
    assert data_combi in TRAINING_TEST_SET_COMBINATIONS, (f"{data_combi} is not in {TRAINING_TEST_SET_COMBINATIONS}")

    return data_combi

def generate_tables(results):
    # one table for each train_test dataset combination
    table_dict = {key: create_empty_dataframe() for key in TRAINING_TEST_SET_COMBINATIONS}
    for index, row in results.iterrows():
        model_train = get_model_specifier_for_table(row['train_set_ai'])  # model the classifier was trained on
        model_test =  get_model_specifier_for_table(row['test_set_ai']) # model the classifier was tested on
        data_combi = get_dataset_combi_specifier_for_table(row['train_set_ai'], row['test_set_ai'], TEST_DATA_SETS)
        accuracy_score = row['accuracy']
        table_dict[data_combi][model_test][model_train] = accuracy_score
    return table_dict


def process_csv(input_file):
    try:
        with open(input_file, 'r') as csv_infile:
            df_results = pd.read_csv(csv_infile)
            df_result_tables = generate_tables(df_results)

            for k, table in df_result_tables.items():
                cur_output = f"{input_file.split('.csv')[0]}_table_{k}.csv"
                table.to_csv(cur_output, index=True, mode='w', header=True)
                print(f"CSV file '{input_file}' processed successfully. Output saved to '{cur_output}'.")


    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate nicely structured tables from result files')
    parser.add_argument('input_file', help='Input CSV path containing the results', type=str)

    args = parser.parse_args()

    process_csv(args.input_file)
