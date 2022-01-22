import torch
import torch.nn as nn
import torchtext
from torchtext.legacy import data
from torchtext.legacy.data import Field, Iterator
import torch.nn.functional as F
import pandas as pd
import random
import re
import time
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
import csv
from sklearn.metrics import f1_score
import torch.nn.utils.prune as prune

# ========================================
#               DATA
# ========================================
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

def spacy_tokenize(x):
    x = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ",
        str(x))
    x = re.sub(r"[ ]+", " ", x)
    x = re.sub(r"\!+", "!", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\?+", "?", x)
    return [tok.text for tok in tokenizer(x) if tok.text != " "]

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

path = ('mnli_data')
domains = ['fiction', 'government', 'slate', 'telephone', 'travel']
domain_name = 'travel'

TEXT = data.Field(lower=True, tokenize = spacy_tokenize, batch_first = True)
LABEL = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
LOGITS = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

fields = {'sentence1': ('premise', TEXT),
          'sentence2': ('hypothesis', TEXT),
          'gold_label': ('label', LABEL),
          'bert_log_0': ('bert_log_0', LOGITS),
          'bert_log_1': ('bert_log_1', LOGITS),
          'bert_log_2': ('bert_log_2', LOGITS)}

train, dev, test = data.TabularDataset.splits(
    path=path,
    train=('mnli_'+domain_name+'_train.csv'),
    validation=('mnli_'+domain_name+'_dev.csv'),
    test=('mnli_'+domain_name+'_test.csv'),
    format='CSV',
    fields=fields
)

#print(vars(train.examples[0]))
#print(vars(test.examples[0]))

TEXT.build_vocab(train, dev, min_freq=2, vectors=torchtext.vocab.Vectors('glove.6B.300d.txt', unk_init=torch.Tensor.normal_))
#LABEL.build_vocab(train)

BATCH_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=BATCH_SIZE, sort=False, device=device)

# ========================================
#               MODEL
# ========================================
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 200
#OUTPUT_DIM = len(LABEL.vocab)
OUTPUT_DIM = 3
DP_RATIO = 0.2
LEARN_RATE = 0.001


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = EMBEDDING_DIM
        self.hidden_size = HIDDEN_DIM
        self.directions = 2
        self.num_layers = 1
        self.concat = 4
        self.device = device
        self.embedding = nn.Embedding(INPUT_DIM, EMBEDDING_DIM)
        self.projection = nn.Linear(self.embed_dim, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=DP_RATIO)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DP_RATIO)

        self.lin1 = nn.Linear(self.hidden_size * self.directions * self.concat, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, OUTPUT_DIM)

        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.out = nn.Sequential(
            self.lin1,
            self.relu,
            self.dropout,
            self.lin2,
            self.relu,
            self.dropout,
            self.lin3
        )

    def forward(self, batch):
        premise_embed = self.embedding(batch.premise)
        hypothesis_embed = self.embedding(batch.hypothesis)

        premise_proj = self.relu(self.projection(premise_embed))
        hypothesis_proj = self.relu(self.projection(hypothesis_embed))

        h0 = c0 = torch.tensor([]).new_zeros(
            (self.num_layers * self.directions, batch.batch_size, self.hidden_size)).to(self.device)

        _, (premise_ht, _) = self.lstm(premise_proj, (h0, c0))
        _, (hypothesis_ht, _) = self.lstm(hypothesis_proj, (h0, c0))

        premise = premise_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)
        hypothesis = hypothesis_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)

        combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), 1)
        return self.out(combined)

model = BiLSTM()
parameters_to_prune = (        # which parameters we want to prune
    (model.embedding, 'weight'),
    (model.projection, 'weight'),
    (model.lstm, 'weight_ih_l0'),
    (model.lstm, 'weight_hh_l0'),
    (model.lstm, 'weight_ih_l0_reverse'),
    (model.lstm, 'weight_hh_l0_reverse'),
    (model.lin1, 'weight'),
    (model.lin2, 'weight'),
    (model.lin3, 'weight'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('The model has ',count_parameters(model), ' trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
for name, param in model.named_parameters():
    if param.requires_grad:
        #param.data
        print(name)

# ========================================
#               Training
# ========================================

def my_loss(logits_student, target, logits_teacher):
    alpha = 1
    criterion_CE = nn.CrossEntropyLoss()
    L_CE = criterion_CE(logits_student, target)
    criterion_KD = torch.nn.KLDivLoss()
    L_KD = criterion_KD(logits_student, logits_teacher)
    loss = (1-alpha)*L_CE + alpha * L_KD
    return loss


def create_bert_dist_tensor(tags_0, tags_1, tags_2):
    batch_len = tags_0.size()
    dist_list = []
    for i in range(batch_len[0]):
        dist_list.append([tags_0[i], tags_1[i], tags_2[i]])
    #F.log_softmax(torch.tensor(dist_list), dim=1)
    #torch.sigmoid(torch.tensor(dist_list))
    return F.log_softmax(torch.tensor(dist_list), dim=1)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARN_RATE)
criterion = nn.CrossEntropyLoss(reduction = 'sum')
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, epoch):
    model.train();
    train_iter.init_epoch()
    prune_at_epoch = [5, 8, 11, 13, 16, 19, 22, 25, 28]

    if epoch in prune_at_epoch:
        print('pruning')
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.1)

    n_correct, n_total, n_loss = 0, 0, 0
    for batch_idx, batch in enumerate(train_iter):
        optimizer.zero_grad()
        answer = model(batch)

        if epoch < 5:
            bert_logits = create_bert_dist_tensor(batch.bert_log_0, batch.bert_log_1, batch.bert_log_2)
            bert_logits = bert_logits.to(device)
            loss = my_loss(answer, batch.label, bert_logits)
        else:
            loss = criterion(answer, batch.label)

        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        n_loss += loss.item()

        loss.backward();
        optimizer.step()
    train_loss = n_loss / n_total
    train_acc = 100. * n_correct / n_total
    return train_loss, train_acc


def validate(model, iterator):
    model.eval();
    test_iter.init_epoch()
    y_pred = []
    y_true = []
    n_correct, n_total, n_loss = 0, 0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            answer = model(batch)
            #loss = criterion(answer, batch.label)
            bert_logits = create_bert_dist_tensor(batch.bert_log_0, batch.bert_log_1, batch.bert_log_2)
            bert_logits = bert_logits.to(device)
            loss = my_loss(answer, batch.label, bert_logits)

            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            n_total += batch.batch_size
            n_loss += loss.item()

            output = answer.detach().cpu().numpy()
            labels = batch.label.to('cpu').numpy()
            output = np.argmax(output, axis=1)
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

        val_loss = n_loss / n_total
        val_acc = 100. * n_correct / n_total
        val_f1 = f1_score(y_true, y_pred, average='weighted')
        return val_loss, val_acc, val_f1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 38

best_valid_loss = float('inf')
training_stats = []
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iter, optimizer, epoch)
    valid_loss, valid_acc, valid_f1 = validate(model, dev_iter)
    test_loss, test_acc, test_f1 = validate(model, test_iter)
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': train_loss,
            'Training Accur': train_acc,
            'Valid. Loss': valid_loss,
            'Valid. Accur.': valid_acc,
            'Valid. F1.': valid_f1,
            'Test. Accur.': test_acc,
            'Test. F1.': test_f1
        })

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    """if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'bilstm-mnli-'+domain_name+'-model.p')"""

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss}, 'bilstm-mnli-'+domain_name+'-model.p')

    print('####### ecpoch: ', epoch,' #######')
    print('train loss: ', train_loss)
    print('train accuracy: ', train_acc)
    print('val loss: ', valid_loss)
    print('val accuracy: ', valid_acc)
    print('test accuracy: ', test_acc)
    print('test f1: ', test_f1)

    # Display floats with two decimal places.
    pd.set_option('precision', 2)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv('training_stats_lstm_'+domain_name+'.csv')

test_loss, test_acc, test_f1 = validate(model, test_iter)
print('test accuracy: ', test_acc)
print('test f1: ', test_f1)
with open('mnli_lstm_test_res_' + domain_name + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['loss', test_loss])
    writer.writerow(['accuracy', test_acc])
    writer.writerow(['f1', test_f1])
