import data
import tagger

#TRAIN_DATA = 'data/train.txt'
TRAIN_DATA = 'data/toy_train.txt'
TAG_SET = 'data/tags_toy.txt'

print('Loading data...')
train_corpus = data.Text(TRAIN_DATA)
#test_corpus = data.Text('data/test.txt')
tag_set = data.TagSet(TAG_SET)
print('Done')

pos_tagger = tagger.POSTagger(tag_set)
print('Training model...')
pos_tagger.fit(train_corpus)
print('Done')

example = 'He is good'.split(' ')
prob, tags = pos_tagger.predict(example)

print('Predict POS tag for:', example)
print('Tags:', tags, 'With probability:', prob)
