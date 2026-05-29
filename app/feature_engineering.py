import pandas as pd

def process_data(df):

    # padroniza colunas em minúsculo
    df.columns = df.columns.str.lower()

    # feature de valor alto
    if 'amount' in df.columns:

        df['high_amount'] = (
            df['amount'] > 2000
        ).astype(int)

    # cria hora baseada no tempo
    if 'time' in df.columns:

        df['hour'] = (
            (df['time'] // 3600) % 24
        )

        # transações de madrugada
        df['night_transaction'] = (
            (df['hour'] >= 0) &
            (df['hour'] <= 5)
        ).astype(int)

    return df