import pandas as pd

if __name__ == '__main__':

    input_file = 'results.csv'
    output_file = 'results_filtered.csv'

    df = pd.read_csv(input_file)

    if 'geonames' not in df.columns:
        raise ValueError("'geonames' is not in the CSV file.")

    df = df[df['geonames'].astype(str).str.contains('geonames', na=False)]

    df.to_csv(output_file, index=False)

    print(f"File saved in: {output_file}")