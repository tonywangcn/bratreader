from __future__ import unicode_literals, print_function
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger
from itertools import islice
import os

from bratreader.repomodel import RepoModel

try:
	unicode
except:
	unicode = str

#spacy NER translation
BRAT_TO_SPACY_KEYS = {
    'Organization':'ORG',
    'Person':'PERSON',
    'Location':'GPE',
    'Tech':'TECHNOLOGY',
    'Job':'JOB',
    'Abbr':'ABBR',
    'Misc':'MISC'
}

def get_annotated_sents(doc, keys):
    ann_sents = []
    for sent in doc.sentences:
        anns = []
        ann_sent=u" ".join([x.form for x in sent.words])
        for word in sent.words:
            if word.annotations:
                for ann in word.annotations:
                    for label, valency in ann.labels.items():
                        anns.append((ann.realspan[0]-sent.start, ann.realspan[1]-sent.start, keys.get(label,label)))
        if anns:
            ann_sents.append((ann_sent,anns))
    return ann_sents

def normalize_tags(nlp, train_data, exclude_tags):
	'''run tagger on training data and
	add tags from nlp to training data
	to normalize training (i.e ensure that
	original NER functionality is preserved)
	if this is not done, tagger "forgets"
	unused tags
	'''
	for raw_text, entity_offsets in train_data:
		doc = nlp(raw_text)
		for ent in doc.ents:
			overlap = False
			if ent.label_ in exclude_tags:
				break
			for entity_offset in entity_offsets:
				#x1 <= y2 && y1 <= x2
				if (ent.start_char <= entity_offset[0] and entity_offset[1] <= ent.end_char):
					overlap = True
					break
			if not overlap:
				entity_offsets.append((ent.start_char, ent.end_char, ent.label_))
		yield (raw_text, entity_offsets)

def train_ner(nlp, train_data, entity_types):
	'''
	param: entity_types is a list of entities included for training (all caps)
	train NER with data formatted as:
	[
	('text', [(begin, end, LABEL),(begin, end, LABEL)]),
	('Who is Shaka Khan?',[(len('Who is '), len('Who is Shaka Khan'), 'PERSON')])
	]
	'''
	# Add new words to vocab.
	for raw_text, _ in train_data:
		doc = nlp.make_doc(raw_text)
		for word in doc:
			_ = nlp.vocab[word.orth]

	# Add unknown entity types to model
	for entity_type in entity_types:
		if entity_type not in nlp.entity.cfg['actions']['1']:
			if not 'extra_labels' in nlp.entity.cfg or 'extra_labels' in nlp.entity.cfg and entity_type not in nlp.entity.cfg['extra_labels']:
				#print(entity_type)
				nlp.entity.add_label(entity_type)

	random.seed(0)

	nlp.entity.model.learn_rate = 0.001
	for itn in range(1000):
		random.shuffle(train_data)
		loss = 0.
		for raw_text, entity_offsets in train_data:
			doc = nlp.make_doc(raw_text)
			gold = GoldParse(doc, entities=entity_offsets)
			# By default, the GoldParse class assumes that the entities
			# described by offset are complete, and all other words should
			# have the tag 'O'. You can tell it to make no assumptions
			# about the tag of a word by giving it the tag '-'.
			# However, this allows a trivial solution to the current
			# learning problem: if words are either 'any tag' or 'ANIMAL',
			# the model can learn that all words can be tagged 'ANIMAL'.
			#for i in range(len(gold.ner)):
				#if not gold.ner[i].endswith('ANIMAL'):
				#    gold.ner[i] = '-'
			nlp.tagger(doc)
			# As of 1.9, spaCy's parser now lets you supply a dropout probability
			# This might help the model generalize better from only a few
			# examples.
			loss += nlp.entity.update(doc, gold, drop=0.5)
		print(loss)
		if loss < 300:
			break
	# This step averages the model's weights. This may or may not be good for
	# your situation --- it's empirical.
	#nlp.end_training()

	return nlp

def save_model(nlp, model_dir):
	model_dir = pathlib.Path(model_dir)
	if not model_dir.exists():
		model_dir.mkdir()
	assert model_dir.is_dir()
	nlp.save_to_directory(model_dir)

def main(data_dir, model_dir=None, exclude_normalize_tags=None, keys={}):
	'''
	data_dir -> path to brat annotation data. searches recursively
	model_dir -> path to save spacy training model
	exclude_normalize_tags -> list of tags to exclude from normalization. If NONE, no normalization is performed.
	keys -> dict translating brat tags to training tags. keys not in dict will be preserved
	'''
	
	r = RepoModel(data_dir, recursive=True, cached=False)

	nlp = spacy.load('en')

	# v1.1.2 onwards
	if nlp.tagger is None:
		print('---- WARNING ----')
		print('Data directory not found')
		print('please run: `python -m spacy.en.download --force all` for better performance')
		print('Using feature templates for tagging')
		print('-----------------')
		nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

	normalized_train_data = []
	excludes = exclude_normalize_tags #we have manually tagged all instances of these

	for key,data in r.documents.items():
		if exclude_normalize_tags:
			normalized_train_data.extend(normalize_tags(nlp, get_annotated_sents(data, keys), excludes))
		else:
			normalized_train_data.extend(get_annotated_sents(data, keys))

	nlp = train_ner(nlp, normalized_train_data, keys.values())

	doc = nlp(u"Hi Adam,\nSounds great to me. I'll send through the QA department. In the invite you through Skype, and we can discuss if Applause is right for you.\nI look forward to it!\nRegards,\nAndrew")
	for word in doc:
		print(word.text, word.tag_, word.ent_type_)

	if model_dir is not None:
		save_model(nlp, model_dir)

if __name__ == '__main__':
	excludes = ["PERSON", "ORG", "GPE"] #have manually tagged these in BRAT, don't need to "normalize"
	data_path = YOUR_BRAT_DATA_PATH
	main(data_dir = data_path, model_dir='ner', exclude_normalize_tags=excludes, keys = BRAT_TO_SPACY_KEYS)