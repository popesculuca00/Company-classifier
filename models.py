import string
import unidecode


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model_const import industry_decoder, Classifier
        
class IndustryPredictor():
    def __init__(self, checkpoint_path="weights/model_10.pth"):

        self.__set_alphabet()

        self.max_size = 128
        self.decoder = industry_decoder
        self.fill_char = "ă"
        
        self.model = Classifier(70, self.max_size, 0, 148)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
        self.model.eval()

    def __set_alphabet(self):
        keys = string.printable[:-5].replace("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "")
        self.alphabet = {key :index+1 for index, key in enumerate(keys)}
        self.alphabet["ă"] = 0
        self.alphabet_size = len(self.alphabet)


    def __decode_preds(self, preds):

        return self.decoder[ torch.argmax(preds).item() ]

    def predict(self, name:str):
        assert isinstance( name, str), "Input must be of type string"

        name = unidecode.unidecode(name)
        name = list (name + str( self.fill_char * (self.max_size - len(name)) ))
        name = list (map( lambda x : self.alphabet.get(x, 0), name ))[:128]
        name = nn.functional.one_hot( torch.LongTensor(name), num_classes=self.alphabet_size )
        name = name.permute(1,0)
        name = name.unsqueeze(0)
        preds = self.model(name.float())
        
        return self.__decode_preds(preds)
