import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'data/'

column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'track_state', 'weather', 'rank', 'lane', 'horse', 'home',
                    'gender', 'age', 'weight', 'rating', 'jockey', 'trainer', 'owner', 'single_odds', 'double_odds'],
    'jockey': ['jockey', 'group', 'birth', 'age', 'debut', 'weight', 'weight_2', 'race_count', 'first', 'second',
               '1yr_count', '1yr_first', '1yr_second'],
    'owner': ['owner', 'reg_horse', 'unreg_horse', 'owned_horse', 'reg_date', '1yr_count', '1yr_first', '1yr_second',
              '1yr_third', '1yr_money', 'race_count', 'first', 'second', 'third', 'owner_money'],
    'trainer': ['trainer', 'group', 'birth', 'age', 'debut', 'race_count', 'first', 'second', '1yr_count', '1yr_first',
                '1yr_second'],
    'horse': ['horse', 'home', 'gender', 'birth', 'age', 'class', 'group', 'trainer', 'owner', 'father', 'mother',
              'race_count', 'first', 'second', '1yr_count', '1yr_first', '1yr_second', 'horse_money', 'rating',
              'price'],
}

# TODO: select columns to use
used_column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'rank', 'lane', 'horse', 'weight', 'jockey', 'trainer',
                    'double_odds'],

    'jockey': ['date', 'jockey', '1yr_count', '1yr_first', '1yr_second'],

    'trainer': ['date', 'trainer', '1yr_count', '1yr_first', '1yr_second'],

    'horse': ['date', 'horse', 'gender', 'age', 'class', '1yr_count', '1yr_first', '1yr_second']
}
overlapped = ['1yr_count', '1yr_first', '1yr_second']
categorical = ['track_length', 'lane', 'class', 'gender']
null_col_mode = ['gender', 'age', 'class']
null_col_mean = ['1yr_count_horse', '1yr_first_horse', '1yr_second_horse', '1yr_count_jockey',
                 '1yr_first_jockey', '1yr_second_jockey', '1yr_count_trainer', '1yr_first_trainer',
                 '1yr_second_trainer']
numeric = ['weight', 'double_odds', 'rate_horse', 'rate_trainer', 'rate_jockey']
foreign_class = ['외4', '외3', '외2', '외1']
domestic_class = ['국6', '국5', '국4', '국3', '국2', '국1']


def load_data():
    df_dict = dict()  # key: data type(e.g. jockey, trainer, ...), value: corresponding dataframe

    for data_type in ['horse', 'jockey', 'trainer', 'race_result']:
        fnames = sorted(os.listdir(DATA_PATH + data_type))
        df = pd.DataFrame()

        # concatenate all text files in the directory
        for fname in fnames:
            tmp = pd.read_csv(os.path.join(DATA_PATH, data_type, fname), header=None, sep=",",
                              encoding='cp949', names=column_name[data_type])

            if data_type != 'race_result':
                date = fname.split('.')[0]
                tmp['date'] = date[:4] + "-" + date[4:6] + "-" + date[-2:]

            df = pd.concat([df, tmp])

        # cast date column to dtype datetime
        df['date'] = df['date'].astype('datetime64[ns]')

        # append date offset to synchronize date with date of race_result data
        if data_type != 'race_result':
            df1 = df.copy()
            df1['date'] += pd.DateOffset(days=2)  # saturday
            df2 = df.copy()
            df2['date'] += pd.DateOffset(days=3)  # sunday
            df = df1.append(df2)

        # select columns to use
        df = df[used_column_name[data_type]]

        # refine column name
        refined_col = []
        for col in df.columns:
            if col in overlapped:
                col = col + '_' + data_type
            refined_col.append(col)

        df.columns = refined_col

        # insert dataframe to dictionary
        df_dict[data_type] = df

    ####### DO NOT CHANGE #######

    df_dict['race_result']['rank'].replace('1', 1., inplace=True)
    df_dict['race_result']['rank'].replace('2', 2., inplace=True)
    df_dict['race_result']['rank'].replace('3', 3., inplace=True)
    df_dict['race_result']['rank'].replace('4', 4., inplace=True)
    df_dict['race_result']['rank'].replace('5', 5., inplace=True)
    df_dict['race_result']['rank'].replace('6', 6., inplace=True)
    df_dict['race_result']['rank'].replace('7', 7., inplace=True)
    df_dict['race_result']['rank'].replace('8', 8., inplace=True)
    df_dict['race_result']['rank'].replace('9', 9., inplace=True)
    df_dict['race_result']['rank'].replace('10', 10., inplace=True)
    df_dict['race_result']['rank'].replace('11', 11., inplace=True)
    df_dict['race_result']['rank'].replace('12', 12., inplace=True)
    df_dict['race_result']['rank'].replace('13', 13., inplace=True)
    df_dict['race_result']['rank'].replace(' ', np.nan, inplace=True)

    # drop rows with rank missing values
    df_dict['race_result'].dropna(subset=['rank'], inplace=True)

    df_dict['race_result']['rank'] = df_dict['race_result']['rank'].astype('int')
    # make a column 'win' that indicates whether a horse ranked within the 3rd place
    df_dict['race_result']['win'] = df_dict['race_result'].apply(lambda x: 1 if x['rank'] < 4 else 0, axis=1)

    #################################

    # TODO: Make Features
    # # example : dummy_variables
    # df_dict['race_result']['gender'].replace({'암': 'F', '수': 'M', '거': 'G'}, inplace=True)
    # df_dict['race_result'] = pd.concat([df_dict['race_result'],
    #                                     pd.get_dummies(df_dict['race_result']['gender'])], axis=1)
    # del df_dict['race_result']['gender']

    # drop duplicated rows (owner is excluded because we didn't use it)
    df_dict['jockey'].drop_duplicates(subset=['date', 'jockey'], inplace=True)
    df_dict['trainer'].drop_duplicates(subset=['date', 'trainer'], inplace=True)

    # merge dataframes (owner is excluded)
    df = df_dict['race_result'].merge(df_dict['horse'], on=['date', 'horse'], how='left')
    df = df.merge(df_dict['jockey'], on=['date', 'jockey'], how='left')
    df = df.merge(df_dict['trainer'], on=['date', 'trainer'], how='left')

    # drop unnecessary columns which are used only for merging dataframes
    df.drop(['horse', 'jockey', 'trainer'], axis=1, inplace=True)

    df.to_csv('df_final.csv')
    return df


def get_data(test_day, is_training):
    if os.path.exists('df_final.csv'):
        print('preprocessed data exists')
        df = pd.read_csv('df_final.csv', index_col=0)
    else:
        print('preprocessed data NOT exists')
        print('loading data')
        df = load_data()

    df['class'] = df['class'].replace('미', df[df['class'].isin(foreign_class)]['class'].mode()[0])  # replace '미' to mode
    df['class'] = df['class'].replace('외미',
                                      df[df['class'].isin(domestic_class)]['class'].mode()[0])  # replace '외미' to mode

    # replace nan to mode
    for col in null_col_mode:
        value = df[col].mode()[0]
        df[col].fillna(value, inplace=True)

    # replace nan to mean
    for col in null_col_mean:
        value = df[col].mean()
        df[col].fillna(value, inplace=True)

    # construct feature called rate = (1yr_first + 1yr_second)/1yr_count for horse, jockey and trainer
    df['rate_horse'] = 0
    df['rate_trainer'] = 0
    df['rate_jockey'] = 0
    for i in ['horse', 'trainer', 'jockey']:
        rate = 'rate_' + i
        count = '1yr_count_' + i
        first = '1yr_first_' + i
        second = '1yr_second_' + i
        temp = df[df[count] > 0].copy()
        temp[rate] = (temp[first] + temp[second]) / temp[count]
        df[df[count] > 0] = temp
        df.drop([count, first, second], axis=1, inplace=True)

    # one-hot-encoding for categorical features : 'track_length', 'lane', 'gender', 'class'
    df_pp = pd.get_dummies(df, columns=categorical)

    # use standardsclaer for same date, same race_num
    scaler = StandardScaler()
    unique_date = df_pp['date'].unique()
    for date in unique_date:
        temp = df_pp[df_pp['date'] == date].copy()
        unique_race = temp['race_num'].unique()
        for race in unique_race:
            temp2 = temp[temp['race_num'] == race].copy()
            temp2[numeric] = temp2[numeric].apply(
                lambda x: scaler.fit_transform(np.array(x).reshape(-1, 1)).reshape(1, -1)[0])
            temp[temp['race_num'] == race] = temp2
        df_pp[df_pp['date'] == date] = temp

    data_set = df_pp

    # select training and test data by test day
    # TODO : cleaning or filling missing value
    training_data = data_set[~data_set['date'].isin(test_day)]
    test_data = data_set[data_set['date'].isin(test_day)]

    # TODO : make your input feature columns

    # select training x and y
    training_y = training_data['win']
    training_x = training_data.drop(['win', 'date', 'race_num', 'rank', 'double_odds'], axis=1)

    # select test x and y
    test_y = test_data['win']
    test_x = test_data.drop(['win', 'date', 'race_num', 'rank', 'double_odds'], axis=1)

    inspect_test_data(test_x, test_day)

    return (training_x, training_y) if is_training else (test_x, test_y)

def inspect_test_data(test_x, test_days):
    """
    Do not fix this function
    """
    df = pd.DataFrame()

    for test_day in test_days:
        fname = os.path.join(DATA_PATH, 'race_result', test_day.replace('-', '') + '.csv')
        tmp = pd.read_csv(fname, header=None, sep=",",
                          encoding='cp949', names=column_name['race_result'])
        tmp.replace(' ', np.nan, inplace=True)
        tmp.dropna(subset=['rank'], inplace=True)

        df = pd.concat([df, tmp])

    # print(test_x.shape[0])
    # print(df.shape[0])

    assert test_x.shape[0] == df.shape[0], 'your test data is wrong!'

def main():
    get_data(['2019-04-20', '2019-04-21'], is_training=True)


if __name__ == '__main__':
    main()
