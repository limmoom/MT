import csv
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, MBartTokenizer
import random
import os
import torch
import time
from torch.utils.data import Dataset

train_data_m = 'EvaHan2023_train_data/train_24_histories_m_utf8.txt'
train_data_c = 'EvaHan2023_train_data/train_24-historoes_c_utf8.txt'

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model_name = "facebook/mbart-large-cc25"
output_dir = './model_save/mbart-large-cc25'
log_dir = './log'

def processdata(filename_m, filename_c):
    with open(filename_m, 'r', encoding='utf-8') as f:
        data_m = [i.strip().split('\n') for i in f.readlines()]
    with open(filename_c, 'r', encoding='utf-8') as g:
        data_c = [i.strip().split('\n') for i in g.readlines()]
    df = pd.DataFrame({'source':data_c, 'target':data_m})
    return df

class CustomDataset(Dataset):
    def __init__(self, data, src_lang, tgt_lang, model_name, with_labels = True):
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.with_labels = with_labels
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.with_labels:
            src = self.data.loc[index,'source']
            tgt = self.data.loc[index,'target']
            batch = self.tokenizer.prepare_seq2seq_batch(src, tgt_texts = tgt, src_lang = self.src_lang, tgt_lang = self.tgt_lang, return_tensors="pt").to(device)
            input_ids = batch["input_ids"].squeeze(0)
            target_ids = batch["labels"].squeeze(0)
            return input_ids, target_ids
        else:
            src = self.data.loc[index,'source']
            batch = self.tokenizer.prepare_seq2seq_batch(src, src_lang = self.src_lang, return_tensors="pt").to(device)
            input_ids = batch["input_ids"].squeeze(0)
            return input_ids

def create_log_file(filename_log):
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    save_filename = f"cc25large_{current_time}.txt"
    save_path = os.path.join(filename_log, save_filename)
    return save_path

def write_log_train(filename_log, batch_idx, epoch, loss):
    with open(filename_log, 'a') as f:
        f.write(f'epoch: {epoch} batch_idx:{batch_idx}, train_loss: {loss}\n')

def write_log_valid(filename_log, batch_idx, epoch, loss):
    with open(filename_log, 'a') as f:
        f.write(f'epoch: {epoch} batch_idx:{batch_idx}, valid_loss: {loss}\n')

class MyModel(nn.Module):
    def __init__(self, model_name, freeze_bert = False):
        super().__init__()
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, labels):
        output = self.model(input_ids, labels=labels)
        return output.loss

    def generate(self, input_ids, labels, decoder_start_token):
        generated_tokens = self.model.generate(input_ids, decoder_start_token_id = self.tokenizer.lang_code_to_id[decoder_start_token])
        generated_sentences = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        ground_truth_sentences = self.tokenizer.batch_decode(labels, skip_special_tokens=True)[0]
        return generated_sentences, ground_truth_sentences

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save(model, optimizer,epoch):
    state_dict = {
        'net' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch' : epoch
    }
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    save_filename = f"model_epoch_{epoch}_{current_time}.pt"
    save_path = os.path.join(output_dir, save_filename)
    torch.save(state_dict, save_path)

def train_valid_test_split(data, test_size=0.3, random_state=None):
    train = data[int(len(data)*test_size):].reset_index(drop=True)
    valid = data[int(len(data)*0.1):int(len(data)*test_size)].reset_index(drop=True)
    test = data[:int(len(data)*0.1)].reset_index(drop=True)
    return train, valid, test

def train_eval(model, optimizer, train_loader, val_loader, test_loader, epochs=50):
    print('start training')
    for epoch in range(epochs):
        model.train()
        print('epoch:', epoch+1)
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader)): 
            if batch_idx % 500 == 1:
                generated_sentences, ground_truth_sentences = model.generate(batch[0], batch[1], 'zh_CN')
                print('generated_sentences:', generated_sentences)
                print('ground_truth_sentences:', ground_truth_sentences)               
            batch = tuple(t.to(device) for t in batch)
            loss = model(batch[0], batch[1])
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 1:
                print(f"Epoch [{epoch + 1}/{epochs} : batch_idx{batch_idx}] - Train Loss: {train_loss/batch_idx:.4f}",flush=True)
                write_log_train(log_dir, batch_idx, epoch, train_loss/batch_idx)
        save(model, optimizer, epoch)
        eval(model, optimizer, val_loader,epoch)
    test(model, test_loader)

def eval(model, optimizer, val_loader,epoch):
    model.eval()
    eval_loss = 0
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if batch_idx %500 == 1:
                generated_sentences, ground_truth_sentences = model.generate(batch[0], batch[1], 'zh_CN')
                print('generated_sentences:', generated_sentences)
                print('ground_truth_sentences:', ground_truth_sentences)
            loss = model(batch[0], batch[1])
            eval_loss += loss.item()
            if batch_idx %500 == 1:
                print(f"batch_idx : {batch_idx}Valid Loss: {eval_loss/batch_idx:.4f}")
                write_log_valid(log_dir, epoch, eval_loss/batch_idx)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if batch_idx % 500 == 1:
                generated_sentences, ground_truth_sentences = model.generate(batch[0], batch[1], 'zh_CN')
                print('generated_sentences:', generated_sentences)
                print('ground_truth_sentences:', ground_truth_sentences)
            loss = model(batch[0], batch[1])
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

'''
test my save function
'''
def test_save(model, optimizer):
    save(model, optimizer)

if __name__ == '__main__':
    set_seed(42)
    log_dir = create_log_file(log_dir)

    df = processdata(train_data_m, train_data_c)
    train, valid, test = train_valid_test_split(df, test_size=0.3, random_state=42)

    train_set = CustomDataset(train, 'ja_XX', 'zh_CN', model_name)
    train_dataset = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    test_set = CustomDataset(test, 'ja_XX', 'zh_CN', model_name)
    test_dataset = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    valid_set = CustomDataset(valid,'ja_XX','zh_CN',model_name)
    valid_dataset =  torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=0)

    model = MyModel(model_name, freeze_bert = False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_eval(model, optimizer, train_dataset, valid_dataset, test_dataset, epochs=25)
    
