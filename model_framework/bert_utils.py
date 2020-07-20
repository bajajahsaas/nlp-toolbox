import torch
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

def convert_sentences_to_bert_inputs(tokenizer, df):
	df['context'] = df['context'].astype(str)
	
	# # only for debugging
	# df = df[:100]

	sentences = df.context.values
	# sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
	sentences = ["[CLS] " + sentence for sentence in sentences]

	if 'label' in df:
		labels = df.label.values
	else:
		# test.sh (inference on unseen data, labels not available)
		labels = None

	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

	logging.debug("Tokenize the first sentence: {}".format(tokenized_texts[0]))

	MAX_LEN = 128

	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
	# default value of padding is 0.0

	attention_masks = []

	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
		seq_mask = [float(i > 0) for i in seq]
		attention_masks.append(seq_mask)

	if labels is not None:
		logging.debug('Labels length: {}'.format(len(labels)))
		logging.debug('Tokenized texts Length: {}'.format(len(tokenized_texts)))
		assert (len(labels) == len(tokenized_texts))

	return input_ids, labels, attention_masks


def flatten_tensor_and_get_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)


def save_plots_models(outDir, train_loss_set, train_acc_set, val_acc_set, model_sd, epoch, epochs):
	if epoch == epochs:
		# writing final models and plots, skip suffix
		epoch = ""
	else:
		epoch = str(epoch + 1)

	plt.figure(figsize=(15, 8))
	plt.title("Training loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(train_loss_set)
	plotFile = outDir + '/loss_plot' + epoch + '.pdf'
	plt.savefig(plotFile)
	logging.info('Saving loss plot in path: {}'.format(plotFile))

	plt.figure(figsize=(15, 8))
	plt.title("Training/Val Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.plot(train_acc_set, color='red', label='Training Acc')
	plt.plot(val_acc_set, color='green', label='Validation Acc')
	plt.legend(['Train Acc', 'Val Acc'], loc='upper left')
	plotFile = outDir + '/acc_plot' + epoch + '.pdf'
	plt.savefig(plotFile)
	logging.info('Saving loss plot in path: {}'.format(plotFile))

	modelPath = outDir + "/model" + epoch + ".pt"
	torch.save(model_sd, modelPath)