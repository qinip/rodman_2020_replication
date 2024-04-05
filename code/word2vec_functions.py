"""
Created on Sun Mar 10 17:05:48 2019
@author: emma

Modified on April 2 2024
@author: eddy, minh
"""

import gensim
import copy
import numpy as np
from gensim.models import Word2Vec
from sklearn.utils import resample

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Code credit due to Ryan Heuser (https://gist.github.com/quadrismegistus/).
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.key_to_index.keys())    # update for Gensim 4
    vocab_m2 = set(m2.wv.key_to_index.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return m1, m2

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)    # update for Gensim 4
    
    
    # Then for each model...
    for m in [m1, m2]:
        # Replace old vectors_norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]                        
        old_arr = m.wv.vectors                                            # update for Gensim 4
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr
        m.wv.norms = np.linalg.norm(m.wv.vectors, axis=1)

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        
        m.wv.index_to_key = common_vocab               # update for Gensim 4
        m.wv.key_to_index = {word: index for index, word in enumerate(common_vocab)}      # no need to create old/new vocab anymore
        
        # Remove words not in common_vocab from key_to_index and index_to_key
        for word in list(m.wv.key_to_index.keys()):
            if word not in common_vocab:
                del m.wv.key_to_index[word]
                if word in m.wv.index_to_key:  # Check if word exists before removing
                    m.wv.index_to_key.remove(word)        
        
    return m1, m2

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
	Code credit due to Ryan Heuser (https://gist.github.com/quadrismegistus/).
	First, intersect the vocabularies (see `intersection_align_gensim` documentation).
	Then do the alignment on the other_embed model.
	Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
	Return other_embed.
	If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
	"""
    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)
    # get the embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()             # update for Gensim 4
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)  # update for Gensim 4
    other_embed.wv.norms = np.linalg.norm(other_embed.wv.vectors, axis=1)
    

    return other_embed


def align_and_produce_new_model(earth_model, moon_sentences):
    ''' Take a base model and align a second model to it, resampling the '''
    ''' sentences from the second model's corpus before modeling.        '''
    sentences = resample(moon_sentences)
    moon_model = Word2Vec(sentences, vector_size = 100, min_count = 0, epochs = 200, 
                     sg = 1, hs = 0, negative = 5, window = 10, workers = 4)
    moon_model.wv.init_sims()
    tidal_lock = smart_procrustes_align_gensim(earth_model, moon_model)
    moon_model = None
    return tidal_lock
    
    
def produce_model_stats(model):
    gender = []
    intl = []
    german = []
    race = []
    afam = []
    while True:
        try:
            gender.append(model.wv.similarity('equality','gender'))
            break
        except KeyError:
            gender.append('NA')
            break
    while True:
        try:
            intl.append(model.wv.similarity('equality','treaty'))
            break
        except KeyError:
            intl.append('NA')
            break
    while True:
        try:
            german.append(model.wv.similarity('equality','german'))
            break
        except KeyError:
            german.append('NA')
            break
    while True:
        try:
            race.append(model.wv.similarity('equality','race'))
            break
        except KeyError:
            race.append('NA')
            break
    while True:
        try:
            afam.append(model.wv.similarity('equality','african_american'))
            break
        except KeyError:
            afam.append('NA')
            break
    stats = gender, intl, german, race, afam
    return list(stats)
    
    
def iterate_model_stats(list_of_lists, iterations):
    full_stats = []
    earth_model = None
    earth_model = Word2Vec(list_of_lists[0], vector_size = 100, min_count = 0, epochs = 200, 
                       sg = 1, hs = 0, negative = 5, window = 10, workers = 4, compute_loss=True)
    # earth_model.wv.init_sims()                 # init_sims() was deprecated in Gensim 4
    for k in range(0, len(list_of_lists)-1):
        era_stats = []
        for i in range(0, iterations):
            earth = copy.deepcopy(earth_model)
            moon_model = align_and_produce_new_model(earth, list_of_lists[k+1])
            iter_stats = produce_model_stats(moon_model)
            era_stats.append(iter_stats)
            run = i+1
            era = k+1
            print("Finished with run %d out of %d for era %d." % (run, iterations, era))
            earth = None
        new_earth = None
        new_earth = Word2Vec(list_of_lists[k], vector_size = 100, min_count = 0, epochs = 200, 
                     sg = 1, hs = 0, negative = 5, window = 10, workers = 4)
        new_moon = Word2Vec(list_of_lists[k+1], vector_size = 100, min_count = 0, epochs = 200, 
                     sg = 1, hs = 0, negative = 5, window = 10, workers = 4)
        earth_model = smart_procrustes_align_gensim(new_earth, new_moon)
        full_stats.append(era_stats)
        print("*******Finished with era %d.*******" % (era))
    return full_stats
    
    
def chrono_train(n_iterations, current_corpus, previous_model, output_model):
    ''' Models the current corpus by initializing with the vectors of  '''
    ''' the previous model, outputs similarity scores of new model,    '''
    ''' and saves that new model for the next round of modeling        '''
    gender = []
    intl = []
    german = []
    race = []
    afam = []
    social = []
    for k in range(n_iterations):
        sentence_samples = resample(current_corpus)
        model = Word2Vec.load(previous_model)
        model.train(sentence_samples, total_examples = len(sentence_samples), epochs = model.epochs)
        gender.append(model.wv.similarity('equality','gender'))
        intl.append(model.wv.similarity('equality','treaty'))
        german.append(model.wv.similarity('equality','german'))
        race.append(model.wv.similarity('equality','race'))
        afam.append(model.wv.similarity('equality','african_american'))
        social.append(model.wv.similarity('equality','social'))
        run = k+1
        print("Finished with run %d out of %d" % (run, n_iterations))
    model.save(output_model)
    stats = gender, intl, german, race, afam, social
    return list(stats) 
    
    
    
    
    