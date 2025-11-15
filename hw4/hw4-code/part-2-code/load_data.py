import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        self.process_data(data_folder, split, self.tokenizer)

    
    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)
    
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
        else:
            sql_lines = [""] * len(nl_lines)   # dummy targets
    
        # -------------------------
        # ❗ TRAIN ON RAW NL INPUT  
        # -------------------------
        processed_nl = nl_lines   # <-- the ONLY correct behavior for training
    
        # Tokenize
        encoder_inputs = tokenizer(
            processed_nl,
            padding=False,
            truncation=True,
            return_attention_mask=True
        )
    
        if split != "test":
            decoder_outputs = tokenizer(
                sql_lines,
                padding=False,
                truncation=True,
                return_attention_mask=True
            )
            self.decoder_ids = decoder_outputs["input_ids"]
            self.decoder_attention = decoder_outputs["attention_mask"]
        else:
            self.decoder_ids = None
            self.decoder_attention = None
    
        self.encoder_ids = encoder_inputs["input_ids"]
        self.encoder_attention = encoder_inputs["attention_mask"]


    
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split == "test":
            return {
        "encoder_ids": torch.tensor(self.encoder_ids[idx]),
        "encoder_mask": torch.tensor(self.encoder_attention[idx])
    }
        else:
            return {
        "encoder_ids": torch.tensor(self.encoder_ids[idx]),
        "encoder_mask": torch.tensor(self.encoder_attention[idx]),
        "decoder_ids": torch.tensor(self.decoder_ids[idx]),
        "decoder_mask": torch.tensor(self.decoder_attention[idx]),
    }


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item["encoder_ids"] for item in batch]
    encoder_masks = [item["encoder_mask"] for item in batch]
    decoder_ids = [item["decoder_ids"] for item in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    decoder_targets = pad_sequence(decoder_ids, batch_first=True, padding_value=PAD_IDX)

    bos_token = torch.tensor([tokenizer.convert_tokens_to_ids("<extra_id_0>")])
    bos_token = bos_token.to(decoder_targets.device)
    bos_column = bos_token.unsqueeze(0).repeat(decoder_targets.size(0), 1)

    decoder_inputs = torch.cat([bos_column, decoder_targets[:, :-1]], dim=1)

    initial_decoder_inputs = bos_column

    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = [item["encoder_ids"] for item in batch]
    encoder_masks = [item["encoder_mask"] for item in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    bos_token = torch.tensor([tokenizer.convert_tokens_to_ids("<extra_id_0>")])
    initial_decoder_inputs = bos_token.unsqueeze(0).repeat(encoder_ids.size(0), 1)

    return encoder_ids, encoder_masks, initial_decoder_inputs



def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))


    return train_x, train_y, dev_x, dev_y, test_x