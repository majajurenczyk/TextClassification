from data import get_data_dict
from numpy import argmax, mean
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def classify(classes, k, classification_mode):
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
        classification_model = make_pipeline(CountVectorizer(stop_words='english'), SelectKBest(chi2, k=k),
                                             MultinomialNB())
    elif classification_mode == 2:
        classification_model = make_pipeline(TfidfVectorizer(stop_words='english'), SelectKBest(chi2, k=k),
                                             LinearSVC(random_state=0))
    else:
        print("mode may be only 1 - NB or 2 - SVM")
        return

    cross_validator = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    metrics = cross_validate(classification_model, reviews_list, classes_list, cv=cross_validator)

    print("TEST SCORE: ",  metrics['test_score'])

    best_accuracy_index = argmax(metrics['test_score'])

    print("BEST ACCURACY INDEX: ",  best_accuracy_index)
    print("BEST ACCURACY: {0:.0%}".format(metrics['test_score'][best_accuracy_index]))
    print("AVG ACCURACY: {0:.0%}".format(mean(metrics['test_score'])))

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
    score = classification_model.score(X_test, Y_test)
    print("SCORE: " + str(score))

    print(classification_model.predict(["Peak filmmaking on the grandest scale and THE most monumentally produced, impeccably designed, and harrowingly epic film I have ever seen. 'Titanic' will never not leave me utterly floored. It's been several months since I last watched this, so I'm trying to remain calm…but it just means so damn much when you cherish the commitment and physical craft a production like this takes and how miraculous it is, not only that Cameron's film turned out this spectacularly, but that we will likely never see an undertaking of this caliber ever again. That deafening mechanical roar when the all the lights go out, like a groaning beast from the deep....chills every time."]))
    print(classification_model.predict(["Mr. Bean's Holiday is better than the first film, but only because it focused more on the character and less on trying to have a plot. The film is ridiculous just as it should be, but there is a creepiness about Bean in some moments that I couldn't get past. Rowan Atkinson is still funny though and still keeps the character the same, its just that at some parts I wondered how weird they were trying to make him. I also still feel that they cannot make a 90 minute movie about Mr. Bean and keep it interesting all the way through, because honestly I was bored at many times. Its pretty much the same as the first film, other than doing the smart thing and getting rid of a story and just doing a series of Bean's trouble makings."]))


if __name__ == '__main__':
    NB = 1
    SVM = 2
    print("=============================================")
    classify(3, 28000, NB)
    print("=============================================")
    classify(4, 10000, NB)
    print("=============================================")
    classify(3, 28000, SVM)
    print("=============================================")
    classify(4, 6000, SVM)
    print("=============================================")





