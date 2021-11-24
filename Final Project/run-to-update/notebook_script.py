import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import json
import subprocess
import glob
import torch

import sys
sys.path.append(root_path)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def create_credentials(username, password, root_path='DataX15/Final Project/'):
  credentials = {"username": username, "password": password}
  jsonString = json.dumps(credentials)
  jsonFile = open(root_path + "run-to-update/secret.json", "w")
  jsonFile.write(jsonString)
  jsonFile.close()


def update_reviews_studied_companies(root_path='DataX15/Final Project/'):
  companies = pd.read_csv(root_path + 'datasets/studied_companies.csv', sep = ';')
  for company in companies.Company:
    # Get company info
    url = companies[companies['Company'] == company].URL.values[0]
    last_date = companies[companies['Company'] == company]['Latest review'].values[0]
    company_file = pd.read_csv(root_path + f'datasets/brut-datasets/{company}_reviews.csv', sep = ';')

    # Prepare and run the terminal command to scrape the missing last reviews and append them to the existing csv file
    command = ['python',  root_path + 'run-to-update/scraper_script.py',  '--headless', '--url', url,
              '--min_date', str(last_date), '-f', root_path + f'datasets/brut-datasets/{company}_reviews.csv']
    subprocess.run(command)

    # Update studied_companies.csv file
    a = pd.read_csv(root_path + f'datasets/brut-datasets/{company}_reviews.csv', sep = ';')
    max_date = max(a.date.values)
    companies.loc[(companies['Company'] == company), 'Latest review'] = max_date
  companies.to_csv(root_path + 'datasets/studied_companies.csv', sep = ';', index = False)


def get_reviews_new_companies(min_date, root_path='DataX15/Final Project/', clean=True):
  if not (pathlib.Path(root_path + 'datasets/new_companies.csv').is_file()) or (os.path.getsize(root_path + 'datasets/new_companies.csv') <= 0):
    return 'No new companies to scrape'
    
  new_companies = pd.read_csv(root_path + 'datasets/new_companies.csv', sep = ';')
  old_companies = pd.read_csv(root_path + 'datasets/studied_companies.csv', sep = ';')

  if new_companies.empty:
    return 'No new companies to scrape'

  for company in new_companies.Company:
    # Get company info
    url = new_companies[new_companies['Company'] == company].URL.values[0]

    # Prepare and run the terminal command to scrape the reviews and create a new csv file for this company
    command = ['python',  root_path + 'run-to-update/scraper_script.py',  '--headless', '--url', url,
              '--min_date', str(min_date), '-f', root_path + f'datasets/brut-datasets/{company}_reviews.csv']
    subprocess.run(command)

    # Update studied_companies.csv file
    a = pd.read_csv(root_path + f'datasets/brut-datasets/{company}_reviews.csv', sep = ';')
    new_row = {'Company': company, 'URL': url, 'Latest review': max(a.date.values)}
    old_companies = old_companies.append(new_row, ignore_index=True)
  old_companies.to_csv(root_path + 'datasets/studied_companies.csv', sep = ';', index = False)

  if clean:
    pd.DataFrame(columns = ['Company', 'URL']).to_csv(root_path + 'datasets/new_companies.csv', sep = ';', index = False)
  return "Done"

def clean_reviews(data):
  data['employee_status'] = ' '.join(data['employee_status'].astype(str).split()[:2])
  data['review_title'] = data['review_title'].astype(str).apply(lambda x: '. '.join(x.split('\n')))
  data['pros'] = data['pros'].astype(str).apply(lambda x: '. '.join(x.split('\n')))
  data['cons'] = data['cons'].astype(str).apply(lambda x: '. '.join(x.split('\n')))
  return data

def assemble_all_reviews(root_path='DataX15/Final Project/'):
  columns = ['Company', 'date', 'employee_title', 'employee_status', 'review_title', 'pros', 'cons']
  all_reviews = pd.DataFrame(columns = columns)
  for f in glob.glob(root_path + "datasets/brut-datasets/*.csv", recursive = True):
    current_reviews = pd.read_csv(f, sep = ';')
    company = f.split('/')[-1][:-12] # get 'COMPANY' from path/to/folder/COMPANY_reviews.csv
    current_reviews['Company'] = company
    all_reviews = all_reviews.append(current_reviews[columns].drop_duplicates())
  all_reviews = clean_reviews(all_reviews)
  all_reviews.to_csv(root_path + 'datasets/all_reviews.csv', index = False, sep = ';')
  return all_reviews

def predict_sentiment(data, root_path='DataX15/Final Project/'):
  analyzer = SentimentIntensityAnalyzer()
  data['score_pros'] = pd.DataFrame(list(data['pros'].apply(analyzer.polarity_scores))).compound
  data['score_cons'] = -pd.DataFrame(list(data['cons'].apply(analyzer.polarity_scores))).compound
  data.to_csv(root_path + 'datasets/all_reviews.csv', index = False, sep = ';')
  return data

def predict_social_classification(data, root_path='DataX15/Final Project/'):
  for model_name in os.listdir(root_path + 'topic-detection/social-models'):
    model = torch.load(model_name)
    criteria = model_name.split('_')[0] # get 'CRITERIA' from CRITERIA_model.extension
    data[f'{criteria}_pros'] = model.predict(data['pros'])
    data[f'{criteria}_cons'] = model.predict(data['cons'])
  data.to_csv(root_path + 'datasets/all_reviews.csv', index = False, sep = ';')
