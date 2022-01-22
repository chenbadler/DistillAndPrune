import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
import time
import datetime
import numpy as np
from sklearn.metrics import f1_score
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv


####----Load The Data for MNLI----####
domains = ['fiction', 'government', 'slate', 'telephone', 'travel']
domain_name = 'travel'
def data_loading(data_path):
    all_sentences = []
    all_labels = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list_of_sent = []
                sent1 = row[1]
                sent2 = row[2]
                gold_label = row[3]
                list_of_sent.append(sent1)
                list_of_sent.append(sent2)
                #all_sentences.append(sent1 + ' ' + sent2)
                all_sentences.append(list_of_sent)
                all_labels.append(gold_label)
                #if gold_label not in tags_vals:
                    #tags_vals.append(gold_label)
                line_count += 1
        print('Processed ' + str(line_count-1) + ' lines.')
    return all_sentences, all_labels

#loading data
class_names = ['entailment', 'contradiction', 'neutral']
tag2idx = {t: i for i, t in enumerate(class_names)}

train_sent, train_lab = data_loading(domain_name+'_train.csv')
train_tags = [tag2idx.get(lab) for lab in train_lab]

dev_sent, dev_lab = data_loading(domain_name+'_dev.csv')
dev_tags = [tag2idx.get(lab) for lab in dev_lab]

test_sent, test_lab = data_loading(domain_name+'_test.csv')
test_tags = [tag2idx.get(lab) for lab in test_lab]


if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Tokenize all of the sentences and map the tokens to their word IDs.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_input_ids = []
train_attention_masks = []
dev_input_ids = []
dev_attention_masks = []
test_input_ids = []
test_attention_masks = []

train_tokenized_texts = [['[CLS]'] + tokenizer.tokenize(sent[0]) + ['[SEP]'] + tokenizer.tokenize(sent[1]) + ['[SEP]'] for sent in train_sent]
train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized_texts],
                          maxlen=128, dtype="long", truncating="post", padding="post")
train_attention_masks = [[float(i > 0) for i in ii] for ii in train_input_ids]

dev_tokenized_texts = [['[CLS]'] + tokenizer.tokenize(sent[0]) + ['[SEP]'] + tokenizer.tokenize(sent[1]) + ['[SEP]'] for sent in dev_sent]
dev_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in dev_tokenized_texts],
                          maxlen=128, dtype="long", truncating="post", padding="post")
dev_attention_masks = [[float(i > 0) for i in ii] for ii in dev_input_ids]

test_tokenized_texts = [['[CLS]'] + tokenizer.tokenize(sent[0]) + ['[SEP]'] + tokenizer.tokenize(sent[1]) + ['[SEP]'] for sent in test_sent]
test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokenized_texts],
                          maxlen=128, dtype="long", truncating="post", padding="post")
test_attention_masks = [[float(i > 0) for i in ii] for ii in test_input_ids]


# Convert the lists into tensors.
train_input_ids = torch.tensor(train_input_ids)
train_attention_masks = torch.tensor(train_attention_masks)
train_labels = torch.tensor(train_tags)

dev_input_ids = torch.tensor(dev_input_ids)
dev_attention_masks = torch.tensor(dev_attention_masks)
dev_labels = torch.tensor(dev_tags)

test_input_ids = torch.tensor(test_input_ids)
test_attention_masks = torch.tensor(test_attention_masks)
test_labels = torch.tensor(test_tags)

# Print sentence 0, now as a list of IDs.
#print('Original: ', train_sent[0])
#print('Token IDs:', train_input_ids[0])

# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset), # Select batches randomly
            batch_size=batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler=SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size=batch_size # Evaluate with this batch size.
        )

test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler=SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size=batch_size # Evaluate with this batch size.
        )


# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=3, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions=False, # Whether the model returns attentions weights.
    output_hidden_states=False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

# Get all of the model's parameters as a list of tuples.
"""params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))"""

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
optimizer = AdamW(model.parameters(),
                  lr=2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5, guy did 1e-4
                  eps=1e-8, # args.adam_epsilon  - default is 1e-8.
                  #weight_decay=0.01
                )

# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default value in run_glue.py
                                            num_training_steps=total_steps)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate():
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)
    test_accur = np.sum(flat_predictions == flat_true_labels) / len(flat_true_labels)
    test_f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
    print('test accuracy: ', test_accur)
    print('f1 test score: ', test_f1)
    with open('mnli_bert_test_res_'+domain_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['accuracy', test_accur])
        writer.writerow(['f1', test_f1])


def create_logits(sent1, sent2):
    tokenized_text = ['[CLS]'] + tokenizer.tokenize(sent1) + ['[SEP]'] + tokenizer.tokenize(sent2) + ['[SEP]']
    input_id = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_text)],
                                    maxlen=128, dtype="long", truncating="post", padding="post")
    attention_mask = [[float(i > 0) for i in ii] for ii in input_id]
    tensor_input = torch.tensor(input_id).to(device)
    tensor_mask = torch.tensor(attention_mask).to(device)

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(tensor_input, token_type_ids=None,
                        attention_mask=tensor_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    return logits[0]


def create_logits_file():
    files = ['train', 'dev', 'test']
    for file_name in files:
        sentences = []
        labels = []
        data = []
        with open(domain_name + '_' + file_name + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    sent1 = row[1]
                    sent2 = row[2]
                    gold_label = row[3]
                    #tow_sent = sent1 + ' ' + sent2
                    #sentences.append(tow_sent)
                    labels.append(gold_label)
                    logits = create_logits(sent1, sent2)
                    data.append((sent1, sent2, tag2idx[gold_label], logits))
                    line_count += 1
            print('Processed ' + str(line_count - 1) + ' lines.')

        with open('mnli_' + domain_name + '_' + file_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['sentence1', 'sentence2', 'gold_label', 'bert_log_0', 'bert_log_1', 'bert_log_2'])
            for sample in data:
                writer.writerow([sample[0], sample[1], sample[2], sample[3][0], sample[3][1], sample[3][2]])

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')
df_stats.to_csv('stats_for_mnli_'+domain_name+'_bert.csv')

# Save model
torch.save(model.state_dict(), 'BERT_MNLI_'+domain_name+'.p')

# Evaluate on test and create data file with bert logits
evaluate()
create_logits_file()