from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import bs4 as bs
import re
import torch
from torch.utils.data import DataLoader


class MYDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    

def BERT_text_cleaner(reviews, nature = 'pros', lemmatize = True, stem = False):
    """
    Clean and preprocess a review.

    Args:
        reviews::[pd.DataFrame]
            The table of given reviews and their statistics.
        nature::[str]
            Should be 'pros' or 'cons' depending on the review we assess
        lemmatize::[boolean]
            A flag for feature lemmatization.
        stem::[boolean]
            A flag for feature stemming.
            
    Return:
        cleaned_reviews::[pd.DataFrame]
            The cleaned version of the reviews.
    """
    
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    
    #1. Remove HTML tags
    cleaned_reviews=[]
    for i,review in enumerate(reviews[nature].astype(str)):
    # print progress
        if (i + 1) % 500 == 0:
            print("Done with %d reviews" %(i+1))
        review = bs.BeautifulSoup(review).text

        #2. Use regex to find emoticons
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)

        #3. Remove punctuation
        review = re.sub("[^a-zA-Z]", " ", review)

        #4. Tokenize into words (all lower case)
        review = review.lower().split()

        #5. Remove stopwords
        eng_stopwords = set(stopwords.words("english"))
            
        clean_review = []
        for word in review:
            if word not in eng_stopwords:
                if lemmatize is True:
                    word = wnl.lemmatize(word)
                elif stem is True:
                    if word == 'oed':
                        continue
                    word = ps.stem(word)
                clean_review.append(word)

        #6. Join the review to one sentence
        review_processed = ' '.join(clean_review + emoticons)
        cleaned_reviews.append(review_processed)
    
    return cleaned_reviews


def open_BERT_model(criteria, root_path = 'DataX15/Final Project/'):
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load(root_path + f'topic-detection/social-models/BERT-models/BERT_{criteria.lower()}.pkl'))
    device = torch.device("cpu")
    model.to(device)
    return model


def predict_class(data, criteria, nature = 'pros', root_path = 'DataX15/Final Project/'):
    # Initialize input
    print('Preparing the input...', end = 'r')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    reviews = pd.DataFrame(BERT_text_cleaner(data, nature, lemmatize = True, stem = False))
    reviews.columns = ['text']
    tokenized_dep = tokenizer(reviews['text'].astype(str).values.tolist(), padding = "max_length", truncation = True)
    dep_labels = [0] * len(reviews.astype(str).values.tolist())
    dep_dataset = MYDataset(tokenized_dep, dep_labels)

    dep_loader = DataLoader(dep_dataset, batch_size = 1, shuffle = False)
    y_predict = []
    count = 1

    # Open the model
    print('Opening the model...', end = 'r')
    model = open_BERT_model(criteria, root_path)
    device = torch.device("cpu")

    # Make the predictions
    print('Making predictions...', end = 'r')
    for batch in dep_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        
        predictions = torch.argmax(outputs.logits, dim = 1)
        y_predict_batch = predictions.cpu().detach().numpy()
        for j in y_predict_batch:
            y_predict.append(j)
            
        del input_ids
        del attention_mask
        del labels
        del outputs
        del predictions
        
        if count % 250 == 0:
            print(f'Batch {count} for review loader completed.')
        count += 1

    y_predict = np.array(y_predict)
    data[f'{criteria}_{nature}'] = y_predict
    print('Done', end = 'r')
    return data