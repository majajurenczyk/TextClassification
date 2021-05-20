from sklearn.feature_extraction.text import CountVectorizer


def read_reviews_to_list():  # READS ALL REVIEWS TO LIST
    with open("scaledata/Dennis+Schwartz/subj.Dennis+Schwartz") as reviews:
        reviews_list = [line.rstrip('\n') for line in reviews]
    with open("scaledata/James+Berardinelli/subj.James+Berardinelli") as reviews:
        reviews_list += [line.rstrip('\n') for line in reviews]
    with open("scaledata/Scott+Renshaw/subj.Scott+Renshaw") as reviews:
        reviews_list += [line.rstrip('\n') for line in reviews]
    with open("scaledata/Steve+Rhodes/subj.Steve+Rhodes") as reviews:
        reviews_list += [line.rstrip('\n') for line in reviews]
    return reviews_list


def read_label3_to_list():  # READS ALL LABEL 3 TO LIST
    with open("scaledata/Dennis+Schwartz/label.3class.Dennis+Schwartz") as lab3:
        reviews_list = [line.rstrip('\n') for line in lab3]
    with open("scaledata/James+Berardinelli/label.3class.James+Berardinelli") as lab3:
        reviews_list += [line.rstrip('\n') for line in lab3]
    with open("scaledata/Scott+Renshaw/label.3class.Scott+Renshaw") as lab3:
        reviews_list += [line.rstrip('\n') for line in lab3]
    with open("scaledata/Steve+Rhodes/label.3class.Steve+Rhodes") as lab3:
        reviews_list += [line.rstrip('\n') for line in lab3]
    return reviews_list


def read_label4_to_list():  # READS ALL LABEL 4 TO LIST
    with open("scaledata/Dennis+Schwartz/label.4class.Dennis+Schwartzz") as lab4:
        reviews_list = [line.rstrip('\n') for line in lab4]
    with open("scaledata/James+Berardinelli/label.4class.James+Berardinelli") as lab4:
        reviews_list += [line.rstrip('\n') for line in lab4]
    with open("scaledata/Scott+Renshaw/label.4class.Scott+Renshaw") as lab4:
        reviews_list += [line.rstrip('\n') for line in lab4]
    with open("scaledata/Steve+Rhodes/label.4class.Steve+Rhodes") as lab4:
        reviews_list += [line.rstrip('\n') for line in lab4]
    return reviews_list


def read_ratings_to_list():  # READS ALL RATINGS TO LIST
    with open("scaledata/Dennis+Schwartz/rating.Dennis+Schwartz") as ratings:
        ratings_list = [line.rstrip('\n') for line in ratings]
    with open("scaledata/James+Berardinelli/rating.James+Berardinelli") as ratings:
        ratings_list += [line.rstrip('\n') for line in ratings]
    with open("scaledata/Scott+Renshaw/rating.Scott+Renshaw") as ratings:
        ratings_list += [line.rstrip('\n') for line in ratings]
    with open("scaledata/Steve+Rhodes/rating.Steve+Rhodes") as ratings:
        ratings_list += [line.rstrip('\n') for line in ratings]
    return ratings_list


def count_reviews_for_label(labels):  # COUNTS HOW MANY REVIEWS LABEL HAS
    result = {}
    for label in labels:
        if label in result:
            result[label] += 1
        else:
            result[label] = 0
    return result


def count_prob_of_label(labels):  # COUNT OCC PROB FOR LABEL
    reviews_for_label = count_reviews_for_label(labels)
    sum_of_reviews = sum(reviews_for_label.values())
    result = {}
    for label in reviews_for_label:
        result[label] = reviews_for_label[label] / sum_of_reviews
    return result


def count_features_occurings_for_label(labels, list_of_reviews):  # COUNTS HOW MANY TIMES FEATURE OCCURS FOR LABELS
    result = {}
    features_for_reviews = convert_reviews_to_features_lists(list_of_reviews)
    for i in range(len(list_of_reviews)):
        if labels[i] not in result:
            result[labels[i]] = {}
        for feature in features_for_reviews[i]:
            if feature not in result[labels[i]]:
                result[labels[i]][feature] = 0
            else:
                result[labels[i]][feature] += 1
    return result


def count_features_prob_for_label(labels, list_of_reviews):
    result = {}
    features_occurings_for_label = count_features_occurings_for_label(labels, list_of_reviews)
    for label in features_occurings_for_label:
        result[label] = {}
        for feature in features_occurings_for_label[label]:
            result[label][feature] = features_occurings_for_label[label][feature]\
                                     / sum(features_occurings_for_label[label].values())
    return result


def convert_reviews_to_features_lists(list_of_reviews):  # CONVERTS LIST OF REVIEWS TO LIST OF LISTS WITH REVIEWS
    # FEATURES
    result = []
    vectorizer = CountVectorizer()
    for rev in list_of_reviews:
        vectorizer.fit_transform([rev])
        result.append(vectorizer.get_feature_names())
    return result


def convert_reviews_to_words_lists(list_of_reviews):  # CONVERTS LIST OF REVIEWS TO LIST OF LISTS WITH REVIEWS
    # WORDS
    result = []
    for rev in list_of_reviews:
        result.append(rev.split())
    return result


def count_number_of_words_in_reviews(list_of_reviews):  # GIVES NUMBER OF WORDS FOR EACH REVIEW IN LIST
    list_of_reviews_by_words = convert_reviews_to_words_lists(list_of_reviews)
    result = []
    for rev in list_of_reviews_by_words:
        result.append(len(rev))
    result.sort(reverse=True)
    return result


def count_number_of_features_in_reviews(list_of_reviews):  # GIVES NUMBER OF FEATURES FOR EACH REVIEW IN LIST
    list_of_reviews_by_features = convert_reviews_to_features_lists(list_of_reviews)
    result = []
    for rev in list_of_reviews_by_features:
        result.append(len(rev))
    result.sort(reverse=True)
    return result


def extract_features_from_text(document):
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(document)
    return vectorizer, matrix


def get_features_names(features):  # GET LIST OF FEATURES FOR ALL REVIEWS
    return features[0].get_feature_names()


def get_features_occurings_dict(features):  # GET DICTIONARY WITH FEATURE AND HOW MANY THIS FEATURE OCCURS IN ALL
    # REVIEWS
    names = get_features_names(features)
    matrix = features[1].toarray()
    sum_of_collumns = matrix.sum(axis=0)
    counter = 0
    result = {}
    while counter < len(names):
        result[names[counter]] = sum_of_collumns[counter]
        counter = counter + 1
    return sorted(result.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    revs = read_reviews_to_list()
    ex_features = extract_features_from_text(revs)
    print("\nFEATURES\n")
    print(get_features_names(ex_features))
    print("\nMATRIX OF FEATURES\n")
    print(ex_features[1].toarray())
    print("\nFEATURES OCCURINGS IN ALL REVIEWS\n")
    print(get_features_occurings_dict(ex_features))
    print("\nNUMBER OF FEATURES FOR EACH REVIEW\n")
    print(count_number_of_features_in_reviews(revs))
    print("\nNUMBER OF WORDS FOR EACH REVIEW\n")
    print(count_number_of_words_in_reviews(revs))

    labs = read_label3_to_list()
    print("\nNUMBER OF REVIEWS IN LABEL\n")
    print(count_reviews_for_label(labs))
    print("\nPROB FOR LABEL OCC\n")
    print(count_prob_of_label(labs))
    print('\nFEATURES OCC IN LABELS\n')
    features_occurings_for_label_m = count_features_occurings_for_label(labs, revs)
    #for el in features_occurings_for_label_m:
     #  print(el + '\n')
     #   print(features_occurings_for_label_m[el])
    #print('\nFEATURES PROBS IN LABELS\n')
    #features_prob_for_label_m = count_features_prob_for_label(labs, revs)
    #for el in features_prob_for_label_m:
        #print(el + '\n')
        #print(features_prob_for_label_m[el])


