'''
check if csv contain asymmetry
'''
import pandas as pd

def check(name) -> bool:
    df = pd.read_csv(name)
    print(df['target'].apply(lambda x: '3' in x).any())
    
check('labels_352x928.csv')