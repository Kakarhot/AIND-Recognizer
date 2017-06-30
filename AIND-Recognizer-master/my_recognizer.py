import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_index in range(test_set.num_items):
        X, lengths = test_set.get_item_Xlengths(word_index)
        prob_dict = {}

        for word, model in models.items():
            try:
                score = model.score(X, lengths)
                prob_dict[word] = score

            except:
                prob_dict[word] = float("-Inf")
                
        probabilities.append(prob_dict)
        guess_word = max(prob_dict, key=prob_dict.get)  
        guesses.append(guess_word)

    return probabilities, guesses
       


