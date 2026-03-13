import pandas as pd
import numpy as np

def optimize_memory(df):
    """Optimise la mémoire par downcasting et affiche le gain."""
    before = df.memory_usage(deep=True).sum() / 1024
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)
    after = df.memory_usage(deep=True).sum() / 1024
    
    print(f'Avant  : {before:.2f} KB')
    print(f'Apres  : {after:.2f} KB')
    print(f'Gain   : {((before - after) / before * 100):.1f}%')
    return df