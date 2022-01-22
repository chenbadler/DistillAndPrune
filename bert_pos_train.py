import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam, BertForSequenceClassification
from seqeval.metrics import f1_score
import numpy as np
from tqdm import tqdm, trange
import csv
import copy

import matplotlib.pyplot as plt
from matplotlib import colors

####----Load The Data for POS----####
data = open("train.wtag", "r")
test = open("test.wtag", "r")
sentences = []
labels = []
tags_vals = []
for line in data:
    sentence = []
    labels_per_sent = []
    splitted = line.split()  # split after space
    for word in splitted:
        word_and_tag = word.split("_")
        sentence.append(word_and_tag[0])
        labels_per_sent.append(word_and_tag[1])
        if word_and_tag[1] not in tags_vals:
            tags_vals.append(word_and_tag[1])
    string_sentence = ' '.join(sentence)
    sentences.append(string_sentence)
    labels.append(labels_per_sent)

for line in test:
    sentence = []
    labels_per_sent = []
    splitted = line.split()  # split after space
    for word in splitted:
        word_and_tag = word.split("_")
        sentence.append(word_and_tag[0])
        labels_per_sent.append(word_and_tag[1])
        if word_and_tag[1] not in tags_vals:
            tags_vals.append(word_and_tag[1])
    string_sentence = ' '.join(sentence)
    sentences.append(string_sentence)
    labels.append(labels_per_sent)

tags_vals.append('none')
#print(sentences[0])
#print(labels[0])
tag2idx = {t: i for i, t in enumerate(tags_vals)}



####----Prepare the sentences and labels----####

"Here we fix some configurations. We will limit our sequence length to 75 tokens and we will use a batch size of 32 " \
"as suggested by the Bert paper. Note, that Bert natively supports sequences of up to 512 tokens. "
#for pos
#MAX_LEN = 128
#for mnli
MAX_LEN = 256
bs = 32
#32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

"The Bert implementation comes with a pretrained tokenizer and a definied vocabulary. We load the one related to the " \
"smallest pre-trained model bert-base-uncased. Try also the cased variate since it is well suited for NER. "
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

"Now we tokenize all sentences"
tokenized_texts = [['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]'] for sent in sentences]
"Next, we cut and pad the token and label sequences to our desired length."
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                        maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                        maxlen=MAX_LEN, value=tag2idx["none"], padding="post",
                        dtype="long", truncating="post")



"The Bert model supports something called attention_mask, which is similar to the masking in keras. So here we create " \
"the mask to ignore the padded elements in the sequences. "
attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

"Now we split the dataset to use 10% to validate the model."
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

"Since we’re operating in pytorch, we have to convert the dataset to torch tensors."
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

"The last step is to define the dataloaders. We shuffle the data at training time with the RandomSampler and at test " \
"time we just pass them sequentially with the SequentialSampler. "
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

####----Setup the Bert model for finetuning----####

"The pytorch-pretrained-bert package provides a BertForTokenClassification class for token-level predictions. " \
"BertForTokenClassification is a fine-tuning model that wraps BertModel and adds token-level classifier on top of the " \
"BertModel. The token-level classifier is a linear layer that takes as input the last hidden state of the sequence. " \
"We load the pre-trained bert-base-uncased model and provide the number of possible labels. "

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

"Now we have to pass the model parameters to the GPU."
model.cuda();

"Before we can start the fine-tuning process, we have to setup the optimizer and add the parameters it should update. " \
"A common choice is the Adam optimizer. We also add some weight_decay as regularization to the main weight matrices. " \
"If you have limited resources, you can also try to just train the linear classifier on top of Bert and keep all " \
"other weights fixed. This will still give you a good performance. "
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
#lr=3e-5


####----Finetune Bert----####
"First we define some metrics, we want to track while training. We use the f1_score from the seqeval package. And we " \
"use simple accuracy on a token level comparable to the accuracy in keras. "

def flat_accuracy(preds, labels):
    #print('pred: ', preds)
    #print('label: ', labels)
    pred_flat = np.argmax(preds, axis=2).flatten()

    #print('preds_max: ', pred_flat)
    labels_flat = labels.flatten()
    labels_len = 0
    sum_true_pred = 0
    for i in range(len(labels_flat)):
        if labels_flat[i] != 44:
            labels_len += 1
            #print('true: ' + str(labels_flat[i]))
            #print('pred: ' + str(pred_flat[i]))
            if labels_flat[i] == pred_flat[i]:
                sum_true_pred += 1
    return sum_true_pred/labels_len
    #print('pred:')
    #print(pred_flat)
    #print('true:')
    #print(labels_flat)
    #return np.sum(pred_flat == labels_flat) / len(labels_flat)


"Finally, we can fine-tune the model. A few epochs should be enough. The paper suggest 3-4 epochs."
epochs = 10
#for pos 33
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    #print('pred tags',pred_tags)
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        #print('valid_tags', valid_tags)
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


####----Evaluate the model----####
def model_eval(model_temp):
    model_temp.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model_temp(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model_temp(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
    valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l]
    loss_temp = eval_loss / nb_eval_steps
    acc_temp = eval_accuracy / nb_eval_steps
    f1_temp = f1_score(pred_tags, valid_tags)
    print("Validation loss: {}".format(loss_temp))
    print("Validation Accuracy: {}".format(acc_temp))
    print("Validation F1-Score: {}".format(f1_temp))
    return loss_temp, acc_temp, f1_temp

model_eval(model)


####----Save the model----####
torch.save(model.state_dict(), 'BERT_POS.p')
