import pickle
import DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np


def main():

    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    test_x, test_y = DataGenerator.get_data(test_day, is_training=False)


    # TODO: fix pickle file name
    filename = 'team05_model.pkl'
    models = pickle.load(open(filename, 'rb'))
    print('load complete')
    for k in models.keys():
        print(k, models[k].get_params())

    ranking_result = np.array([22, 16, 2, 26, 3, 32, 42, 44, 41, 43, 28, 23, 34, 21, 12, 5, 4,
                               8, 6, 7, 9, 10, 31, 39, 25, 20, 19, 18, 11, 1, 24, 35, 36, 13,
                               29, 30, 38, 40, 27, 17, 15, 33, 37, 14])

    feature_number_list = [25, 30, 40]
    features = test_x.columns
    feature_set_list = []
    for num in feature_number_list:
        feature = features[ranking_result < num]
        feature_set_list.append(feature)
    # ================================ predict result ========================================
    pred_y = list()
    index=0
    for k in models.keys():
        pred_y.append(models[k].predict(test_x[feature_set_list[index]]))
        index=index+1
    pred_y = np.ceil(np.mean(pred_y, axis=0))

    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}'.format(f1_score(test_y, pred_y)))



if __name__ == '__main__':
    main()