import data
import tagger

TRAIN_DATA = 'data/train.txt'
#TRAIN_DATA = 'data/toy_train.txt'

print('Loading data...')
train_corpus = data.Text(TRAIN_DATA)
#test_corpus = data.Text('data/test.txt')
print('Done')

pos_tagger = tagger.POSTagger()
print('Training model...')
pos_tagger.fit(train_corpus)
print('Done')

example = 'You lie to me right?'.split(' ')
prob, tags = pos_tagger.predict(example)

print('Predict POS tag for:', example)
print('Tags:', tags, 'With probability:', prob)
