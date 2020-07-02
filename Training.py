from sklearn import svm
import pickle
import DataGenerator
import numpy as np


def main():
    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

    # ================================ train SVM model=========================================
    # TODO: set parameters
    print('start training model')

    # Determine feature ranking by RFE
    ranking_result = np.array([22, 16, 2, 26, 3, 32, 42, 44, 41, 43, 28, 23, 34, 21, 12, 5, 4,
                               8, 6, 7, 9, 10, 31, 39, 25, 20, 19, 18, 11, 1, 24, 35, 36, 13,
                               29, 30, 38, 40, 27, 17, 15, 33, 37, 14])

    # final two feature_set, model parameters (selected by grid search and ensemble)
    feature_number_list = [25, 30, 40]
    features = training_x.columns
    feature_set_list = []
    for num in feature_number_list:
        feature = features[ranking_result < num]
        feature_set_list.append(feature)
    models = dict()
    models['model with 25 features'] = svm.SVC(C=1, kernel='linear', class_weight='balanced', random_state=42)
    models['model with 30 features']= svm.SVC(C=1, kernel='linear', class_weight='balanced', random_state=42)
    models['model with 40 features'] = svm.SVC(C=1, kernel='linear', class_weight='balanced', random_state=42)

    np.random.seed(42)
    # index is for feature_set_list
    index = 0
    for k in models.keys():
        choosed_indices = np.random.choice(training_x.index, len(training_x), replace=True)
        training_x_choosed = training_x.loc[choosed_indices, feature_set_list[index]]
        training_y_choosed = training_y.loc[choosed_indices]
        models[k].fit(training_x_choosed, training_y_choosed)
        index = index + 1

    print('completed training model')

    # TODO: fix pickle file name
    filename = 'team05_model.pkl'
    pickle.dump(models, open(filename, 'wb'))
    print('save complete')


if __name__ == '__main__':
    main()
