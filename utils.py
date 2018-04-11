import numpy as np

def createVocabulary(input_path, output_path, no_pad=False):
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'r') as fd, \
         open(output_path, 'w+') as out:
        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()

            for w in words:
                if w == '_UNK':
                    break
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ['_PAD', '_UNK'] + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ['_UNK'] + sorted(vocab, key=vocab.get, reverse=True)

        for v in vocab:
            out.write(v+'\n')

def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}

def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if str.isdigit(w) == True:
            w = '0'
        ids.append(vocab.get(w, vocab['_UNK']))

    return ids

def padSentence(s, max_length, vocab):
    return s + [vocab['vocab']['_PAD']]*(max_length - len(s))

# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart = False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd = False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd

def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType

def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                     __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                     (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100*correctChunkCnt/foundCorrectCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return f1, precision, recall

class DataProcessor(object):
    def __init__(self, in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab):
        self.__fd_in = open(in_path, 'r')
        self.__fd_slot = open(slot_path, 'r')
        self.__fd_intent = open(intent_path, 'r')
        self.__in_vocab = in_vocab
        self.__slot_vocab = slot_vocab
        self.__intent_vocab = intent_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_slot.close()
        self.__fd_intent.close()

    def get_batch(self, batch_size):
        in_data = []
        slot_data = []
        slot_weight = []
        length = []
        intents = []

        batch_in = []
        batch_slot = []
        max_len = 0

        #used to record word(not id)
        in_seq = []
        slot_seq = []
        intent_seq = []
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            slot = self.__fd_slot.readline()
            intent = self.__fd_intent.readline()
            inp = inp.rstrip()
            slot = slot.rstrip()
            intent = intent.rstrip()

            in_seq.append(inp)
            slot_seq.append(slot)
            intent_seq.append(intent)

            iii=inp
            sss=slot
            inp = sentenceToIds(inp, self.__in_vocab)
            slot = sentenceToIds(slot, self.__slot_vocab)
            intent = sentenceToIds(intent, self.__intent_vocab)
            batch_in.append(np.array(inp))
            batch_slot.append(np.array(slot))
            length.append(len(inp))
            intents.append(intent[0])
            if len(inp) != len(slot):
                print(iii,sss)
                print(inp,slot)
                exit(0)
            if len(inp) > max_len:
                max_len = len(inp)

        length = np.array(length)
        intents = np.array(intents)
        #print(max_len)
        #print('A'*20)
        for i, s in zip(batch_in, batch_slot):
            in_data.append(padSentence(list(i), max_len, self.__in_vocab))
            slot_data.append(padSentence(list(s), max_len, self.__slot_vocab))
            #print(s)
        in_data = np.array(in_data)
        slot_data = np.array(slot_data)
        #print(in_data)
        #print(slot_data)
        #print(type(slot_data))
        for s in slot_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            slot_weight.append(weight)
        slot_weight = np.array(slot_weight)
        return in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq
