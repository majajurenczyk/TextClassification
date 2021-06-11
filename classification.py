from data import get_data_dict
from numpy import argmax, mean
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import time


def classify(classes, k, classification_mode, alpha, C, min_df, max_df):
    start = time.time()

    data_dict = get_data_dict()
    reviews_list = data_dict['review']

    if classes == 3:
        classes_list = data_dict['label3']
    elif classes == 4:
        classes_list = data_dict['label4']
    else:
        print("classes may be only 3 or 4")
        return

    if classification_mode == 1:
        #   cv = CountVectorizer(stop_words='english', max_df=max_df, min_df=min_df)
        cv = CountVectorizer(stop_words='english')
        classification_model = make_pipeline(cv, SelectKBest(chi2, k=k),
                                             MultinomialNB(alpha=alpha, fit_prior=True))
                                             #ComplementNB())
                                             #GaussianNB())
                                             #BernoulliNB())
                                             #CategoricalNB())
    elif classification_mode == 2:
        classification_model = make_pipeline(TfidfVectorizer(stop_words='english'), SelectKBest(chi2, k=k),
                                             LinearSVC(random_state=0, C=C))
    else:
        print("mode may be only 1 - NB or 2 - SVM")
        return

    cross_validator = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    metrics = cross_validate(classification_model, reviews_list, classes_list, cv=cross_validator)

    #   print("TEST SCORE: ",  metrics['test_score'])

    best_accuracy_index = argmax(metrics['test_score'])

    #   print("BEST ACCURACY INDEX: ",  best_accuracy_index)
    #   print("BEST ACCURACY: {0:.0%}".format(metrics['test_score'][best_accuracy_index]))
    #   print("AVG ACCURACY: {0:.0%}".format(mean(metrics['test_score'])))

    list_index_gen = list(cross_validator.split(reviews_list, classes_list))

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for i in list_index_gen[best_accuracy_index][0]:
        X_train.append(reviews_list[i])
        Y_train.append(classes_list[i])

    for i in list_index_gen[best_accuracy_index][1]:
        X_test.append(reviews_list[i])
        Y_test.append(classes_list[i])

    classification_model.fit(X_train, Y_train)
    res_score = classification_model.score(X_test, Y_test)
    #   print("SCORE: " + str(score))
    return res_score

    #   print(classification_model.predict(["The film is ridiculous just as it should be, but there is a creepiness about  some moments that I couldn't get past. Rowan Atkinson is still funny though and still keeps the character the same, its just that at some parts I wondered how weird they were trying to make him. I also still feel that they cannot make a 90 minute movie about Mr. Bean and keep it interesting all the way through, because honestly I was bored at many times. Its pretty much the same as the first film, other than doing the smart thing and getting rid of a story and just doing a series of Bean's trouble makings."]))


def hiperparams_test():
    for i in range(1, 11):
        startNB = time.time()
        scoreNB = classify(3, 20000, 1, float(i / 10), float(i / 2))
        endNB = time.time()
        timeNB = endNB - startNB

        startSVM = time.time()
        scoreSVM = classify(3, 20000, 2, float(i / 10), float(i / 2))
        endSVM = time.time()
        timeSVM = endSVM - startSVM

        print(str(scoreNB).replace('.', '.') + '\t' + str(timeNB).replace('.', '.') + '\t' +
              str(scoreSVM).replace('.', '.') + '\t' + str(timeSVM).replace('.', '.'))

    for i in range(3, 9):
        startNB = time.time()
        scoreNB = classify(3, 20000, 1, float(i / 2), float(i / 2))
        endNB = time.time()
        timeNB = endNB - startNB

        startSVM = time.time()
        scoreSVM = classify(3, 20000, 2, float(i / 2), float(i / 2))
        endSVM = time.time()
        timeSVM = endSVM - startSVM

        print(str(scoreNB).replace('.', ',') + '\t' + str(timeNB).replace('.', ',') + '\t' +
              str(scoreSVM).replace('.', ',') + '\t' + str(timeSVM).replace('.', ','))


def k_param_test():
    for i in range(1, 11):
        startNB = time.time()
        scoreNB = classify(3, int(float(i / 10) * 35000), 1, 1, 0.4)
        endNB = time.time()
        timeNB = endNB - startNB

        startSVM = time.time()
        scoreSVM = classify(3, int(float(i / 10) * 35000), 2, 1, 0.4)
        endSVM = time.time()
        timeSVM = endSVM - startSVM

        print(str(scoreNB).replace('.', ',') + '\t' + str(timeNB).replace('.', ',') + '\t' +
              str(scoreSVM).replace('.', ',') + '\t' + str(timeSVM).replace('.', ','))


def k_param_test1():
    for i in range(1, 8):
        startNB = time.time()
        scoreNB = classify(3, int(i * 5000), 1, 1, 0.4)
        endNB = time.time()
        timeNB = endNB - startNB

        startSVM = time.time()
        scoreSVM = classify(3, int(i * 5000), 2, 1, 0.4)
        endSVM = time.time()
        timeSVM = endSVM - startSVM

        print(str(scoreNB).replace('.', ',') + '\t' + str(timeNB).replace('.', ',') + '\t' +
              str(scoreSVM).replace('.', ',') + '\t' + str(timeSVM).replace('.', ','))


def labels_test():
    for i in range(3, 5):
        startNB = time.time()
        scoreNB = classify(i, 28000, 1, 1, 0.1, 1, 1.0)
        endNB = time.time()
        timeNB = endNB - startNB

        startSVM = time.time()
        scoreSVM = classify(i, 28000, 2, 1, 0.1, 1, 1.0)
        endSVM = time.time()
        timeSVM = endSVM - startSVM

        print(str(scoreNB).replace('.', ',') + '\t' + str(timeNB).replace('.', ',') + '\t' +
              str(scoreSVM).replace('.', ',') + '\t' + str(timeSVM).replace('.', ','))


def min_max_test():
    for i in range(0, 4):
        score_mm = classify(3, 0, 1, 1, 0.4, float(i / 10), 1.0)  # MIN MAX
        print(str(score_mm).replace('.', ','))
    for i in range(7, 10):
        score_mm = classify(3, 0, 1, 1, 0.4, 1, float(i / 10))
        print(str(score_mm).replace('.', ','))

    score_mm = classify(3, 0, 1, 1, 0.4, 0.1, 0.6)
    print(str(score_mm).replace('.', ','))

    score_mm = classify(3, 0, 1, 1, 0.4, 0.1, 0.8)
    print(str(score_mm).replace('.', ','))


if __name__ == '__main__':
    #   for i in range(3, 9):
    #   print(str(i/2))

    NB = 1
    SVM = 2
    #   print("=============================================")
    #   classify(3, 0, NB, 1, 1)
    #   print("=============================================")
    #   classify(4, 10000, NB)
    #   print("=============================================")
    print(classify(3, 28000, NB, 1, 0.4, 1, 1.0))
    print(classify(4, 28000, NB, 1, 0.4, 1, 1.0))
    #   print("=============================================")
    #   classify(4, 6000, SVM)
    #   print("=============================================")
    #   hiperparams_test()
    #   k_param_test()
    #   min_max_test()

    #   score = classify(3, 0, 1, 1, 0.4, 0.02, 1.0)
    #   print(str(score).replace('.', ','))
    #labels_test()
