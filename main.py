import pandas as pd

# define targets
SEQ_LEN = 60                    # use last 60 minutes
FUTURE_PERIOD_PREDICT = 3       # to predict future 3 minutes
RATIO_TO_PREDICT = 'LTC-USD'    # predict on Litecoin


# make targets
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


main_df = pd.DataFrame()
ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

# get and prepare ds
for ratio in ratios:
    dataset = f'crypto_data/{ratio}.csv'  # path to csv files

    df = pd.read_csv(
        dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])

    # give ratio and volume individual column_names
    df.rename(columns={'close': f'{ratio}_close',
                       'volume': f'{ratio}_volume'}, inplace=True)

    df.set_index('time', inplace=True)

    # only use ratio and volume as ds
    df = df[[f'{ratio}_close', f'{ratio}_volume']]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# get future prize vs close prize
main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(
    -FUTURE_PERIOD_PREDICT)
#print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future']].head())

# get info about if buy or !buy
main_df['target'] = list(
    map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
# print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head())
