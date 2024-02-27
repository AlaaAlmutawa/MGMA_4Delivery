## reference: https://medium.com/@cgawande12/unraveling-the-web-a-deep-dive-into-the-google-web-graph-dataset-43604786ac31

## import

import pandas as pd

def load(path):
    # Load the dataset
    data = pd.read_csv(path, sep='\t', comment='#', names=['Source', 'Target'])
    # Basic statistics and information about the dataset
    info = {
        "Number of Rows": len(data),
        "Number of Unique Sources": data['Source'].nunique(),
        "Number of Unique Targets": data['Target'].nunique(),
        "Number of Unique Nodes": len(set(data['Source']).union(set(data['Target']))),
        "Missing Values in Source": data['Source'].isnull().sum(),
        "Missing Values in Target": data['Target'].isnull().sum()
    }
    print(info)
    return data

def reduce(data,frac,random_state=42):
    # Reduce the dataset to 50% of its size
    reduced_data = data.sample(frac=frac, random_state=random_state)

    # Display the size of the reduced dataset
    reduced_data_size = len(reduced_data)
    print('Reduced dataset size:', reduced_data_size)
    info = {
        "Number of Rows": len(reduced_data),
        "Number of Unique Sources": reduced_data['Source'].nunique(),
        "Number of Unique Targets": reduced_data['Target'].nunique(),
        "Number of Unique Nodes": len(set(reduced_data['Source']).union(set(reduced_data['Target']))),
        "Missing Values in Source": reduced_data['Source'].isnull().sum(),
        "Missing Values in Target": reduced_data['Target'].isnull().sum()
    }
    print('*'*10 + 'info' + '*'*10 + '\n')
    print(info)
    return reduced_data

if __name__ == "__main__":
    # Load the dataset
    data = load('data/web-Google.txt')
    # Reduce the dataset to 50% of its size
    reduced_data = reduce(data,0.5)
    # Save the reduced dataset to a new file
    reduced_data.to_csv('data/web-Google-reduced.txt', sep='\t', index=False, header=False)
    print('Reduced dataset saved to data/web-Google-reduced.txt')


