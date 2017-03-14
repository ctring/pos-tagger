import data
import tagger

TRAIN_DATA = 'data/train.txt'

print('Loading data...')
train_corpus = data.Text(TRAIN_DATA)
print('Done')

pos_tagger = tagger.POSTagger()
print('Training model...')
pos_tagger.fit(train_corpus)
pos_tagger.save_model('model.pickle')
print('Done')

example = 'I am God'.split(' ')
prob, tags = pos_tagger.predict(example)

print('Predict POS tag for:', example)
print('Tags:', tags, 'With probability:', prob)
