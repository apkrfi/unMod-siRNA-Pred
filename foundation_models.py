from multimolecule import RnaTokenizer, RiNALMoModel, RnaFmModel, ErnieRnaModel, RnaBertModel
import pandas as pd

def RinaLMO(sequences):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-mega")
    model = RnaBertModel.from_pretrained("multimolecule/rinalmo-mega")
    features=[]
    for seq in sequences:
        input = tokenizer(seq, return_tensors="pt")
        output = model(**input)
        features.append(output[1][0].detach().numpy())
        
    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])


def RNA_FM(sequences):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaBertModel.from_pretrained("multimolecule/rnafm")
    features=[]
    for seq in sequences:
        input = tokenizer(seq, return_tensors="pt")
        output = model(**input)
        features.append(output)
        
    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])

def RNA_Ernie(sequences):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnaernie")
    model = RnaBertModel.from_pretrained("multimolecule/rnaernie")
    features=[]
    for seq in sequences:
        input = tokenizer(seq, return_tensors="pt")
        output = model(**input)
        features.append(output[1][0].detach().numpy())
        
    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])

def RNA_BERT(sequences):
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
    model = RnaBertModel.from_pretrained("multimolecule/rnabert")
    features=[]
    for seq in sequences:
        input = tokenizer(seq, return_tensors="pt")
        output = model(**input)
        features.append(output[1][0].detach().numpy())

    return pd.DataFrame(features, index=[f"seq{i+1}" for i in range(len(sequences))])

