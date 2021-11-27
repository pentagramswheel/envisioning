import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pathlib
import os
import json
import subprocess
import glob
import torch

import nltk
nltk.download('punkt')

import sys
root_path = 'DataX15/Final Project/'
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
  data['employee_status'] = data['employee_status'].apply(lambda x: ' '.join(nltk.word_tokenize(x)[:2]))
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
  data['score_pros'] = pd.DataFrame(list(data['pros'].astype(str).apply(analyzer.polarity_scores))).compound
  data['score_cons'] = -pd.DataFrame(list(data['cons'].astype(str).apply(analyzer.polarity_scores))).compound
  data.to_csv(root_path + 'datasets/all_reviews.csv', index = False, sep = ';')
  return data

def predict_social_classification(data, root_path='DataX15/Final Project/'):
  for model_name in os.listdir(root_path + 'topic-detection/social-models'):
    model = torch.load(model_name)
    criteria = model_name.split('_')[0] # get 'CRITERIA' from CRITERIA_model.extension
    data[f'{criteria}_pros'] = model.predict(data['pros'])
    data[f'{criteria}_cons'] = model.predict(data['cons'])
  data.to_csv(root_path + 'datasets/all_reviews_S.csv', index = False, sep = ';')


def get_score(data, columns, nature='pros'):
  if nature not in ['pros', 'cons']:
    return 'nature must be either "pros" or "cons"'

  output = pd.DataFrame(index = data['Company'].unique())
  for col in columns:
    score = data[data[f'{col.lower()}_{nature}'] == 1][['Company', f'score_{nature}']]
    score = score.groupby('Company').agg(['mean', 'count']).round(2)
    score.columns = [f'{col}_{nature}_mean', f'{col}_{nature}_count']
    bool_sign = (nature == 'pros') - 1 # 0 for cons, 1 for pros
    score[f'{col}_{nature}_mean'] = MinMaxScaler(feature_range=(100*bool_sign, 100*(bool_sign + 1))).fit_transform(score[f'{col}_{nature}_mean'].values.reshape(-1,1)) # [0, 100] for pros, [-100, 0] for cons
    score[f'{col}_{nature}_count'] = score[f'{col}_{nature}_count']
    score[f'{col}_{nature}_count'] = score[f'{col}_{nature}_count']
    output = output.merge(score, left_index = True, right_index = True, how = 'left').fillna(0)

  total = data[['Company', f'score_{nature}']]
  total = total.groupby('Company').agg(['mean', 'count']).round(2)
  total.columns = [f'Total_{nature}_mean', f'Total_{nature}_count']
  total[f'Total_{nature}_mean'] = MinMaxScaler(feature_range=(100*bool_sign, 100*(bool_sign + 1))).fit_transform(total[f'Total_{nature}_mean'].values.reshape(-1,1))
  output = output.merge(total, left_index = True, right_index = True, how = 'left')
  return output.astype(int)


def aggregate_company_results(data, columns):
  results_pros = get_score(data, columns, 'pros')
  results_cons = get_score(data, columns, 'cons')
  results_aggregated = pd.DataFrame(index = results_pros.index)
  for col in columns + ['Total']:
    results_aggregated[f'{col}_mean'] = (results_pros[f'{col}_pros_mean']*results_pros[f'{col}_pros_count'] + results_cons[f'{col}_cons_mean']*results_cons[f'{col}_cons_count'])/(1 + results_pros[f'{col}_pros_count'] + results_cons[f'{col}_cons_count'])
    results_aggregated[f'{col}_mean'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(results_aggregated[f'{col}_mean'].values.reshape(-1,1))
    results_aggregated[f'{col}_count'] = results_pros[f'{col}_pros_count'] + results_cons[f'{col}_cons_count']
  return results_aggregated.merge(results_pros, left_index = True, right_index = True).merge(results_cons, left_index = True, right_index = True).astype(int)


def get_tableau_input(data, columns, nature=None):
  """
  nature:
  - None if combine averaged scores on pros and cons
  - 'pros' if combine pros scores
  - 'cons' if combine cons scores
  """
  if nature is None:
    extension = ''
  else:
    extension = '_' + nature

  output = pd.DataFrame()


  for col1 in columns:
    x_mean = data[col1 + extension + '_mean'].reset_index()
    x_mean.columns = ['Company', col1]
    new1_mean = pd.melt(x_mean, id_vars=['Company'], value_vars = col1)
    new1_mean.columns = ['Company', 'Criteria 1', 'Score 1']

    x_count = data[col1 + extension + '_count'].reset_index()
    x_count.columns = ['Company', col1]
    new1_count = pd.melt(x_count, id_vars=['Company'], value_vars = col1)
    new1_count.columns = ['Company', 'Criteria 1', 'Count 1']
    
    new1 = new1_mean.merge(new1_count, left_on = ['Company', 'Criteria 1'], right_on = ['Company', 'Criteria 1'])

    for col2 in columns:
      x_mean = data[col2 + extension + '_mean'].reset_index()
      x_mean.columns = ['Company', col2]
      new2_mean = pd.melt(x_mean, id_vars=['Company'], value_vars = col2)
      new2_mean.columns = ['Company', 'Criteria 2', 'Score 2']

      x_count = data[col2 + extension + '_count'].reset_index()
      x_count.columns = ['Company', col2]
      new2_count = pd.melt(x_count, id_vars=['Company'], value_vars = col2)
      new2_count.columns = ['Company', 'Criteria 2', 'Count 2']
      
      new2 = new2_mean.merge(new2_count, left_on = ['Company', 'Criteria 2'], right_on = ['Company', 'Criteria 2'])

      new = pd.merge(new1, new2, left_on = 'Company', right_on = 'Company')
      output = output.append(new)

  rank = output[['Company', 'Score 1']].groupby('Company').mean().rank(ascending = False).astype(int)
  rank.columns = ['Rank']
  output = output.merge(rank.reset_index(), left_on = 'Company', right_on = 'Company')

  output.to_csv(root_path + f'datasets/Tableau_input_S{extension}.csv', index = False, sep = ';')

  return output


def generate_tableau_inputs(data, columns):
  return get_tableau_input(data, columns, None), get_tableau_input(data, columns, 'pros'), get_tableau_input(data, columns, 'cons')