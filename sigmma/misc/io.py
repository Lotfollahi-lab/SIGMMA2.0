import pandas as pd

def save_dict_to_dataframe(dictionary, file_name):
    df = pd.DataFrame.from_dict(
        dictionary,
        orient='index'
    )
    df.index.name = 'n_neighbors'
    df = df.reset_index()
    df.to_csv(file_name, index=False)

def load_dataframe_to_dict(file_name):
    
    df = pd.read_csv(file_name)
    dict_effect_of_n_neighbors = (
        df
        .set_index('n_neighbors')
        .to_dict(orient='index')
    )
    return dict_effect_of_n_neighbors