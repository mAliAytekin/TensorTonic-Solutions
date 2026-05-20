import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    selected = df[column]

    return {
        'values':selected.tolist(),
        'length':int(len(selected))
    }