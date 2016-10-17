import collections
import itertools
import math
import time

import nltk

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        brown_words_s = [START_SYMBOL] * 2
        brown_tags_s = [START_SYMBOL] * 2
        for token in sentence.strip().split():
            slash_index = token.rfind('/')
            word, tag = token[:slash_index], token[slash_index + 1:]
            brown_words_s.append(word)
            brown_tags_s.append(tag)
        brown_words_s.append(STOP_SYMBOL)
        brown_tags_s.append(STOP_SYMBOL)
        brown_words.append(brown_words_s)
        brown_tags.append(brown_tags_s)
    return brown_words, brown_tags

def calc_trigrams(brown_tags):
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)

    for sentence in brown_tags:
        for bigram in nltk.bigrams(sentence):
            bigram_c[bigram] += 1

    for sentence in brown_tags:
        for trigram in nltk.trigrams(sentence):
            trigram_c[trigram] += 1

    trigram_p = {k: math.log(float(v) / bigram_c[k[:2]], 2) for k, v in trigram_c.iteritems()}
    return trigram_p


# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()

def calc_known(brown_words):
    known_words = set()
    words_c = collections.defaultdict(int)
    # Count all the words
    for sentence in brown_words:
        for word in sentence:
            words_c[word] += 1

    # Return non-rare words.
    for word, c in words_c.iteritems():
        if c > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

def replace_rare(brown_words, known_words):
    for i, sentence in enumerate(brown_words):
        for j, word in enumerate(sentence):
            if word not in known_words:
                brown_words[i][j] = RARE_SYMBOL
    return brown_words

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    e_values_c = collections.defaultdict(int)
    tags_c = collections.defaultdict(int)

    for word_sentence, tag_sentence in zip(brown_words_rare, brown_tags):
        for word, tag in zip(word_sentence, tag_sentence):
            e_values_c[(word, tag)] += 1
            tags_c[tag] += 1

    for (word, tag), p in e_values_c.iteritems():
        e_values[(word, tag)] = math.log(float(p) / tags_c[tag], 2)

    return e_values, set(tags_c)

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    pi = collections.defaultdict(float)
    bp = {}
    bp[(-1, START_SYMBOL, START_SYMBOL)] = START_SYMBOL
    pi[(-1, START_SYMBOL, START_SYMBOL)] = 0.0

    for tokens_orig in brown_dev_words:
        tokens = [w if w in known_words else RARE_SYMBOL for w in tokens_orig]

        # k = 1 case
        for w in taglist:
            word_tag = (tokens[0], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            pi[(0, START_SYMBOL, w)] = pi[(-1, START_SYMBOL, START_SYMBOL)] + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
            bp[(0, START_SYMBOL, w)] = START_SYMBOL

        # k = 2 case
        for w in taglist:
            for u in taglist:
                word_tag = (tokens[1], u)
                trigram = (START_SYMBOL, w, u)
                pi[(1, w, u)] = pi.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
                bp[(1, w, u)] = START_SYMBOL

        # k >= 2 case
        for k in range(2, len(tokens)):
            for u in taglist:
                for v in taglist:
                    max_prob = float('-Inf')
                    max_tag = ''
                    for w in taglist:
                        score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + e_values.get((tokens[k], v), LOG_PROB_OF_ZERO)
                        if (score > max_prob):
                            max_prob = score
                            max_tag = w
                    bp[(k, u, v)] = max_tag
                    pi[(k, u, v)] = max_prob

        max_prob = float('-Inf')
        v_max, u_max = None, None
        # finding the max probability of last two tags
        for (u, v) in itertools.product(taglist, taglist):
            score = pi.get((len(tokens_orig) - 1, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if score > max_prob:
                max_prob = score
                u_max = u
                v_max = v
        # append tags in reverse order
        tags = []
        tags.append(v_max)
        tags.append(u_max)

        for count, k in enumerate(range(len(tokens_orig) - 3, -1, -1)):
            tags.append(bp[(k + 2, tags[count + 1], tags[count])])

        tagged_sentence = []
        # reverse tags
        tags.reverse()
        # stringify tags paired with word without start and stop symbols
        for k in range(0, len(tokens_orig)):
            tagged_sentence += [tokens_orig[k], "/", str(tags[k]), " "]
        tagged_sentence.append('\n')
        tagged.append(''.join(tagged_sentence))

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i], brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for sentence in brown_dev_words:
        tagged.append(' '.join([word + '/' + tag for word, tag in trigram_tagger.tag(sentence)]) + '\n')
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__":
    main()
