# Owner: Data-X Team 15
# Adapted version from Matthew Chatham's work (https://github.com/MatthewChatham/glassdoor-review-scraper)

# IMPORTANT: Read README.md for a better idea on how to run this. 
import time
import pandas as pd
from argparse import ArgumentParser
import argparse
import logging
import logging.config
from selenium import webdriver as wd
from selenium.webdriver import ActionChains
import selenium
import numpy as np
from schema import SCHEMA
import json
import urllib
import datetime as dt
import pathlib

start = time.time()
cols = ['date', 'employee_title', 'employee_status', 'review_title', 'pros', 'cons']

DEFAULT_URL = ('https://www.glassdoor.com/Overview/Working-at-'
               'Premise-Data-Corporation-EI_IE952471.11,35.htm')

parser = ArgumentParser()
parser.add_argument('-u', '--url',
                    help='URL of the company\'s Glassdoor landing page.',
                    default=DEFAULT_URL)
parser.add_argument('-f', '--file', default='glassdoor_ratings.csv',
                    help='Output file.')
parser.add_argument('--headless', action='store_true',
                    help='Run Chrome in headless mode.')
parser.add_argument('--username', help='Email address used to sign in to GD.')
parser.add_argument('-p', '--password', help='Password to sign in to GD.')
parser.add_argument('-c', '--credentials', help='Credentials file')
parser.add_argument('-l', '--limit', default=25,
                    action='store', type=int, help='Max pages to scrape')
parser.add_argument('-s', '--start_page', default=1,
                    action='store', type=int, help='Strarting page for reviews')
parser.add_argument('--start_from_url', action='store_true',
                    help='Start scraping from the passed URL.')
parser.add_argument(
    '--max_date', help='Latest review date to scrape.\
    Only use this option with --start_from_url.\
    You also must have sorted Glassdoor reviews ASCENDING by date.',
    type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
parser.add_argument(
    '--min_date', help='Earliest review date to scrape.\
    Only use this option with --start_from_url.\
    You also must have sorted Glassdoor reviews DESCENDING by date.',
    type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
args = parser.parse_args()

if not args.start_from_url and (args.max_date or args.min_date):
    raise Exception(
        'Invalid argument combination:\
        No starting url passed, but max/min date specified.'
    )
elif args.max_date and args.min_date:
    raise Exception(
        'Invalid argument combination:\
        Both min_date and max_date specified.'
    )

if args.credentials:
    with open(args.credentials) as f:
        d = json.loads(f.read())
        args.username = d['username']
        args.password = d['password']
else:
    try:
        with open('secret.json') as f:
            d = json.loads(f.read())
            args.username = d['username']
            args.password = d['password']
    except FileNotFoundError:
        msg = 'Please provide Glassdoor credentials.\
        Credentials can be provided as a secret.json file in the working\
        directory, or passed at the command line using the --username and\
        --password flags.'
        raise Exception(msg)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(lineno)d\
    :%(filename)s(%(process)d) - %(message)s')
ch.setFormatter(formatter)

logging.getLogger('selenium').setLevel(logging.CRITICAL)
logging.getLogger('selenium').setLevel(logging.CRITICAL)


def scrape(field, review, author):

    def scrape_date(review):
        date = author.find_element_by_class_name('authorJobTitle').text
        print(date)
        date = date.split(' -')[0]
        res = pd.to_datetime(date, format = '%b %d, %Y')
        return res

    def scrape_emp_title(review):
        if 'Anonymous Employee' not in review.text:
            try:
                res = author.find_element_by_class_name(
                    'authorJobTitle').text.split(' - ')[1]
            except Exception:
                logger.warning('Failed to scrape employee_title')
                res = "N/A"
        else:
            res = "Anonymous"
        return res
    

    def scrape_status(review):
        """
        try:
            res = author.text.split('-')[0]
        except Exception:
            logger.warning('Failed to scrape employee_status')
            res = "N/A"
        return res
        """
        return review.find_element_by_class_name('eg4psks0').text

    def scrape_rev_title(review):
        return review.find_element_by_class_name('reviewLink').text #.strip('"') #Olivier


    def expand_show_more(section):
        try:
            more_link = section.find_element_by_class_name('v2__EIReviewDetailsV2__continueReading')
            more_link.click()
        except Exception:
            pass

    def scrape_pros(review):
        try:
            pros = review.find_element_by_class_name('gdReview')
            expand_show_more(pros)
            pro_index = pros.text.find('Pros')
            con_index = pros.text.find('Cons')
            res = pros.text[pro_index+5 : con_index]
        except Exception:
            res = np.nan
        return res

    def scrape_cons(review):
        try:
            cons = review.find_element_by_class_name('gdReview')
            expand_show_more(cons)
            con_index = cons.text.find('Cons')
            continue_index = cons.text.find('Continue reading')
            res = cons.text[con_index+5 : continue_index]
        except Exception:
            res = np.nan
        return res


    funcs = [
        scrape_date,
        scrape_emp_title,
        scrape_status,
        scrape_rev_title,
        scrape_pros,
        scrape_cons]


    fdict = dict((s, f) for (s, f) in zip(cols, funcs))

    return fdict[field](review)


def extract_from_page():

    def is_featured(review):
        try:
            review.find_element_by_class_name('featuredFlag')
            return True
        except selenium.common.exceptions.NoSuchElementException:
            return False

    def extract_review(review):
        try:
            author = review.find_element_by_class_name('authorInfo')
        except:
            return None # Account for reviews that have been blocked
        res = {}
        # import pdb;pdb.set_trace()
        for field in cols:
            res[field] = scrape(field, review, author)

        assert set(res.keys()) == set(cols)
        return res

    logger.info(f'Extracting reviews from page {page[0]}')

    res = pd.DataFrame([], columns=cols)

    reviews = browser.find_elements_by_class_name('empReview')
    logger.info(f'Found {len(reviews)} reviews on page {page[0]}')
    
    # refresh page if failed to load properly, else terminate the search
    if len(reviews) < 1:
        browser.refresh()
        time.sleep(5)
        reviews = browser.find_elements_by_class_name('empReview')
        logger.info(f'Found {len(reviews)} reviews on page {page[0]}')
        if len(reviews) < 1:
            valid_page[0] = False # make sure page is populated

    for review in reviews:
        if not is_featured(review):
            data = extract_review(review)
            if data != None:
                logger.info(f'Scraped data for "{data["review_title"]}"\
    ({data["date"]})')
                res.loc[idx[0]] = data
            else:
                logger.info('Discarding a blocked review')
        else:
            logger.info('Discarding a featured review')
        idx[0] = idx[0] + 1

    if args.max_date and \
        (pd.to_datetime(res['date']).max() > args.max_date) or \
            args.min_date and \
            (pd.to_datetime(res['date']).min() < args.min_date):
        logger.info('Date limit reached, ending process')
        date_limit_reached[0] = True

    return res


def more_pages():
    try:
        current = browser.find_element_by_class_name('selected')
        pages = browser.find_element_by_class_name('pageContainer').text.split()
        if int(pages[-1]) != int(current.text):
            return True
        else:
            return False
    except selenium.common.exceptions.NoSuchElementException:
        return False


def go_to_next_page():
    logger.info(f'Going to page {page[0] + 1}')
    next_ = browser.find_element_by_class_name('nextButton')

    ActionChains(browser).click(next_).perform()
    time.sleep(5) # wait for ads to load
    page[0] = page[0] + 1


def no_reviews():
    return False
    # TODO: Find a company with no reviews to test on


def sign_in():
    logger.info(f'Signing in to {args.username}')

    url = 'https://www.glassdoor.com/profile/login_input.htm'
    browser.get(url)

    # import pdb;pdb.set_trace()

    email_field = browser.find_element_by_xpath(
      "//*[@type='submit']//preceding::input[2]"
  )
    password_field = browser.find_element_by_xpath(
      "//*[@type='submit']//preceding::input[1]"
  )
    submit_btn = browser.find_element_by_xpath('//button[@type="submit"]')

    email_field.send_keys(args.username)
    password_field.send_keys(args.password)
    submit_btn.click()

    time.sleep(3)
    browser.get(args.url)



def get_browser():
    logger.info('Configuring browser')
    chrome_options = wd.ChromeOptions()
    if args.headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('log-level=3')
    browser = wd.Chrome(options=chrome_options)
    return browser


def get_current_page():
    logger.info('Getting current page number')
    current = browser.find_element_by_class_name('paginationFooter').text.split(' ')[1]
    return 1 + int(current) // 10


def verify_date_sorting():
    logger.info('Date limit specified, verifying date sorting')
    ascending = urllib.parse.parse_qs(
        args.url)['sort.ascending'] == ['true']

    if args.min_date and ascending:
        raise Exception(
            'min_date required reviews to be sorted DESCENDING by date.')
    elif args.max_date and not ascending:
        raise Exception(
            'max_date requires reviews to be sorted ASCENDING by date.')

def generate_links(url, pages, start_page):
	links = []
	if start_page ==1:
		links.append(url)
		pages -=1
		start_page +=1

	spliced = url[:-4]
	for i in range(pages):
		end = ".htm?filter.iso3Language=eng"

		new_link = spliced + "_P" + str(start_page + i) + end

		links.append(new_link)

	return links

browser = get_browser()
page = [1]
idx = [0]
date_limit_reached = [False]
valid_page = [True]
links = generate_links(args.url, args.limit, args.start_page)



def main():
	
    logger.info(f'Scraping up to {args.limit} Pages.')
    print(links[:2], 'until', links[-1:])
    res = pd.DataFrame([], columns=cols)
    if not pathlib.Path(args.file).is_file():
        res.to_csv(args.file, index=False, sep = ';')

    #sign_in()
    for i, link in enumerate(links):
        browser.get(link)
        time.sleep(5)
        page[0] = get_current_page()
        logger.info(f'Starting from page {page[0]:,}.')
        time.sleep(1)
        if i%1 == 0:
            print(f'{i}-th page: Appending {len(res)} reviews to file {args.file}')
            res.to_csv(args.file, mode='a', index=False, header=False, sep = ';')
            res = pd.DataFrame([], columns=cols)
        reviews_df = extract_from_page()
        res = res.append(reviews_df)
        


    logger.info(f'Writing {len(res)} reviews to file {args.file}')
    res.to_csv(args.file, mode='a', index=False, header=False, sep = ';')

    end = time.time()
    logger.info(f'Finished in {end - start} seconds')


if __name__ == '__main__':
    main()
