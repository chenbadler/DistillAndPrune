import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import numpy as np
import time
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim


####----Load The Data----####
def get_tag2idx():
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
    tag2idx = {t: i for i, t in enumerate(tags_vals)}
    return tag2idx


####----Load teacher----####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
tag2idx = get_tag2idx()
bert = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
bert.load_state_dict(torch.load('bert_pos_934.p'))
bert.eval()


####----Student----####
#mode can be prune (for training with iterative pruning) or train (regular tarinig)
mode = 'train'
start_time = time.time()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def my_loss(logits_student, target, logits_teacher):
    logits_teacher = logits_teacher[:target.size()[0]]
    logits_teacher = logits_teacher.cuda()
    logits_student = logits_student.cuda()
    target = target.cuda()

    soft_max = nn.Softmax(dim=1)

    alpha = 1
    criterion_CE = nn.CrossEntropyLoss()
    L_CE = criterion_CE(logits_student, target)
    criterion_KD = torch.nn.KLDivLoss()
    L_KD = criterion_KD(logits_student, logits_teacher)
    loss = (1-alpha)*L_CE + alpha * L_KD
    return loss
    #return L_KD


def get_bert_logits(sentences):
    MAX_LEN = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert.cuda();
    bert.eval()
    all_logits = []
    for sentence in sentences:
        #string_sentence = ' '.join(sentence)
        tokenized_texts = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_texts)],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        tensor_inputs = torch.tensor(input_ids)
        tensor_masks = torch.tensor(attention_masks)
        with torch.no_grad():
            s_input_ids = tensor_inputs.to(device)
            s_input_mask = tensor_masks.to(device)
            logits = bert(s_input_ids, token_type_ids=None, attention_mask=s_input_mask)
            logits = logits.detach().cpu().numpy()
        logits = logits[0]
        all_logits.append(logits)

    #all_logits = all_logits.detach().cpu().numpy()
    return all_logits


def acc_LSTM(epoch):
    with torch.no_grad():
        true_pred_cnt = 0
        pred_cnt = 0
        for sentence, tags in test_data:
            test_inputs = prepare_sequence(sentence, word_to_ix)
            tag_scores = LSTM(test_inputs)
            word_index = 0
            for test_line in tag_scores:
                test_line = test_line.numpy()
                predicted_tag = np.argmax(test_line)
                pred_cnt += 1
                if predicted_tag == tag2idx[tags[word_index]]:
                    true_pred_cnt += 1
                word_index += 1
        print('epoch: ', str(epoch))
        print('test score for LSTM: ', str(true_pred_cnt / pred_cnt))
        return true_pred_cnt / pred_cnt

data = open("train.wtag", "r")
test = open("test.wtag", "r")
training_data = []
all_tags = []
all_words = []
sentences_for_bert = []
labels_for_bert = []
word_ix = 0
word_to_ix = {}
for line in data:
    temp_sentence = []
    labels_per_sent = []
    splitted = line.split()  # split after space
    for word in splitted:
        word_and_tag = word.split("_")
        temp_sentence.append(word_and_tag[0])
        labels_per_sent.append(word_and_tag[1])

        if word_and_tag[0] not in all_words:
            all_words.append(word_and_tag[0])
    string_sentence = ' '.join(temp_sentence)
    sentences_for_bert.append(string_sentence)
    labels_for_bert.append(labels_per_sent)
    training_data.append((temp_sentence, labels_per_sent))

test_data = []
for line in test:
    temp_sentence = []
    labels_per_sent = []
    splitted = line.split()  # split after space
    for word in splitted:
        word_and_tag = word.split("_")
        temp_sentence.append(word_and_tag[0])
        labels_per_sent.append(word_and_tag[1])
        if word_and_tag[0] not in all_words:
            all_words.append(word_and_tag[0])
        if word_and_tag[1] not in all_tags:
            all_tags.append(word_and_tag[1])
    test_data.append((temp_sentence, labels_per_sent))
all_tags.append('none')
word_to_ix = {t: i for i, t in enumerate(all_words)}


# ========================================
#               MODEL
# ========================================
# Dimension possible sizes: 6 / 32 /  64
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        #print(sentence)
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

LSTM = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag2idx))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(LSTM.parameters(), lr=0.1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('The model has ',count_parameters(LSTM), ' trainable parameters')



# ========================================
#               Training
# ========================================
#0.1 lr
if mode == 'prune':
    checkpoint = torch.load('LSTM_kd_1_100_epoch.p')
    LSTM.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #LSTM.eval()
    # - or -
    LSTM.train()

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = LSTM(inputs)

accuracy_list = []
num_of_epochs = []
prune_at_epoch = [0, 9, 19, 29, 39, 49, 59, 69, 79]  # on which epoch we want to prune
parameters_to_prune = (         # which parameters we want to prune
    (LSTM.word_embeddings, 'weight'),
    (LSTM.lstm, 'weight_ih_l0'),
    (LSTM.lstm, 'weight_hh_l0'),
    (LSTM.hidden2tag, 'weight'),
)
all_logits = get_bert_logits(sentences_for_bert)
for epoch in range(150):
    if mode == 'prune':
        if epoch == 0:
            print('accuracy before pruning: ')
            accuracy_list.append(acc_LSTM(epoch))
        if epoch in prune_at_epoch:
            print('pruning')
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.1,
            )
    i = 0
    num_of_epochs.append(epoch)
    for sentence, tags in training_data:
        bert_logits = all_logits[i]
        bert_logits = torch.from_numpy(bert_logits)

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        LSTM.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag2idx)

        # Step 3. Run our forward pass.
        tag_scores = LSTM(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        #loss_LSTM = my_loss(tag_scores, targets, bert_logits)

        loss_LSTM = loss_function(tag_scores, targets)
        loss_LSTM.backward()
        optimizer.step()
        i += 1
    accuracy_list.append(acc_LSTM(epoch))

    accuracy_list_string = []
    for acc in accuracy_list:
        accuracy_list_string.append(str(acc))

    with open('results_file_LSTM_pruned_1_kd_150_epochs.csv', mode='w') as results_file_LSTM:
        results_writer = csv.writer(results_file_LSTM, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(accuracy_list_string)

    torch.save({'epoch': epoch,
                'model_state_dict': LSTM.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_LSTM}, 'LSTM_pruned_kd_1_150_epoch.p')

print("--- training took %s seconds ---" % (time.time() - start_time))



