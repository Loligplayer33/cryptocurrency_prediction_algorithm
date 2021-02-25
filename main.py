import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

SEQ_LEN = 60                    # use last 60 minutes
FUTURE_PERIOD_PREDICT = 3       # to predict future 3 minutes
RATIO_TO_PREDICT = 'LTC-USD'    # predict on Litecoin


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    #! scale and shuffle df
    df = df.drop('future', 1)
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)  # ? why? just in case!

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])  # * get all values !target
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    #! balance df
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)  # ? shuffle for good measure!

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)  # ? extract buys from sells

    #! create ds for training (x, y)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


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

# get info about if buy or !buy
main_df['target'] = list(
    map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

#! create out of sample ds to prevent overfitting
times = sorted(main_df.index.values)
last_5_pct = times[-int(0.05 * len(times))]

# * seperate out the out-of-sample-data
validation_main_df = main_df[main_df.index >= last_5_pct]
main_df = main_df[main_df.index < last_5_pct]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(
    f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
