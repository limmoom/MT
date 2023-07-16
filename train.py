import time

import os
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, MBartTokenizer, \
    get_linear_schedule_with_warmup

source = "./preprocessed/"
src_train_file = source + "train.en_XX"
tgt_train_file = source + "train.ja_XX"
src_valid_file = source + "valid.en_XX"
tgt_valid_file = source + "valid.ja_XX"
src_test_file = source + "test.en_XX"
tgt_test_file = source + "test.ja_XX"

config = {
    "max_sequence_length": 512,
    "batch_size": 1,
    "epoch": 10,
    "learning_rate": 1e-5,
    "warmup_steps": 4000,
    'save_dir': './models',
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_texts, self.tgt_texts = self.read_files()

    def read_files(self):
        src_texts = []
        tgt_texts = []

        with open(self.src_file, 'r', encoding='utf-8') as src_f, open(self.tgt_file, 'r',
                                                                       encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                src_texts.append(src_line.strip())
                tgt_texts.append(tgt_line.strip())

        return src_texts, tgt_texts

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, index):
        src_text = self.src_texts[index]
        tgt_text = self.tgt_texts[index]

        return src_text, tgt_text


def create_translation_datasets(src_train_file, tgt_train_file, src_valid_file, tgt_valid_file, src_test_file,
                                tgt_test_file, max_sequence_length, batch_size):
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ja_XX")

    train_dataset = TranslationDataset(src_train_file, tgt_train_file, tokenizer, max_sequence_length)
    valid_dataset = TranslationDataset(src_valid_file, tgt_valid_file, tokenizer, max_sequence_length)
    test_dataset = TranslationDataset(src_test_file, tgt_test_file, tokenizer, max_sequence_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def create_dataloader():
    max_sequence_length = config['max_sequence_length']
    batch_size = config['batch_size']

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = create_translation_datasets(
        src_train_file, tgt_train_file, src_valid_file, tgt_valid_file, src_test_file, tgt_test_file,
        max_sequence_length, batch_size
    )
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


class TranslationModel(nn.Module):
    def __init__(self):
        super(TranslationModel, self).__init__()
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)

        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",

                                                              src_lang="en_XX",
                                                              tgt_lang="ja_XX")

    def generate(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(device)
        translated_tokens = self.model.generate(**inputs,
                                                decoder_start_token_id=self.tokenizer.lang_code_to_id["ja_XX"])
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def forward(self, src_text, tgt_text):
        model_inputs = self.tokenizer(src_text, padding=True, return_tensors="pt").to(device)
        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(tgt_text, padding=True, return_tensors="pt").to(device).input_ids
        outputs = self.model(**model_inputs, labels=tgt)

        return outputs[0]


def train(model, train_dataloader, test_dataloader, valid_dataloader):
    epochs = config['epoch']
    total_steps = len(train_dataloader) * epochs

    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'],
                                                num_training_steps=total_steps)

    for epoch in range(epochs):

        # ========================train========================
        model.train()
        train_loss = 0.0
        for train_batch in tqdm(train_dataloader):
            src, tgt = train_batch
            # print(src)
            # print(tgt)
            # count += 1
            loss = model(src, tgt)
            # if count == 8:
            #     return
            # Backward and optimize

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}")

        # ========================test=========================
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for test_batch in tqdm(test_dataloader):
                src, tgt = test_batch
                loss = model(src, tgt)

                test_loss += loss.item()

        test_loss /= len(test_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Test Loss: {test_loss:.4f}")

        # ========================valid========================
        model.eval()

        with torch.no_grad():
            count = 0
            for valid_batch in tqdm(valid_dataloader):
                src_batch, tgt_batch = valid_batch
                prediction = model.generate(src_batch)
                print("Source Text:", src_batch)
                print("Gold Truth:", tgt_batch)
                print("Prediction:", prediction)
                print()
                count += 1
                if count == 8:
                    break

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Epoch [{epoch + 1}/{epochs}] - Current Time: {current_time}")
        # save_model
        save_dir = config['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        save_filename = f"model_epoch_{epoch}_{current_time}.pt"
        save_path = os.path.join(save_dir, save_filename)

        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    print("Training finished.")


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = create_dataloader()
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
    model = TranslationModel()
    # model = None
    train(model, train_loader, test_loader, valid_loader)
