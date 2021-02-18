import pandas as pd
import os
import ast

DATA_DIR = 'data'

REVIEW_FILES = {
    'clothing': 'reviews_Clothing_Shoes_and_Jewelry_5.csv',
    'electronics': 'reviews_Electronics_5.csv',
    'movies': 'reviews_Movies_and_TV_5.csv',
    'books': 'reviews_Books_5.csv'
}

METADATA_FILES = {
    'clothing': 'metadata_category_clothing_shoes_and_jewelry_only.csv',
    'all': 'metadata.csv'
}

# reviewer fields
REVIEWER_ID = 'reviewerID'
PRODUCT_ID = 'asin'
REVIEWER_NAME = 'reviewerName'
HELPFULNESS = 'helpful'
REVIEW_TEXT = 'reviewText'
RATING = 'overall'
SUMMARY = 'summary'
UNIX_REVIEW_TIME = 'unixReviewTime'
REVIEW_TIME = 'reviewTime'

# product fields
SALES_RANK = 'salesRank'
CATEGORIES = 'categories'
PRODUCT_TITLE = 'title'
PRODUCT_DESCRIPTION = 'description'
PRICE = 'price'
RELATED_PRODUCTS = 'related'
BRAND = 'brand'

# inferred fields
BOUGHT_TOGETHER = 'bought_together'
SUMMARY_LENGTH = 'summaryLength'
CATEGORY = 'category'
REVIEW_WORD_COUNT = 'reviewTextLength'
NUM_HELPFUL = 'numHelpful'
NUM_UNHELPFUL = 'numUnhelpful'
NUM_HELPFULNESS_VOTES = 'numHelpfulnessVotes'
PRODUCT_DESCRIPTION_LENGTH = 'productDescriptionLength'

DISPLAY_NAMES_DICT = {
    REVIEWER_ID: 'Reviewer ID',
    PRODUCT_ID: 'Product ID',
    REVIEWER_NAME: 'Reviewer Name',
    HELPFULNESS: 'Fraction of Helpful Votes',
    RATING: 'Rating',
    SALES_RANK: 'Sales Rank',
    PRODUCT_TITLE: 'Product Title',
    PRICE: 'Price',
    BRAND: 'Brand',
    CATEGORY: 'Product Category',
    REVIEW_WORD_COUNT: 'Review Word Count',
    NUM_HELPFUL: 'Number of Helpful Votes',
    NUM_UNHELPFUL: 'Number of Unhelpful Votes',
    NUM_HELPFULNESS_VOTES: 'Total # of Helpfulness Votes'
}

REVIEW_COLUMNS = [REVIEWER_ID,
                  PRODUCT_ID,
                  REVIEWER_NAME,
                  HELPFULNESS,
                  REVIEW_TEXT,
                  RATING,
                  SUMMARY,
                  UNIX_REVIEW_TIME,
                  REVIEW_TIME]

PRODUCT_COLUMNS = [PRODUCT_ID,
                   SALES_RANK,
                   PRODUCT_TITLE,
                   PRODUCT_DESCRIPTION,
                   PRICE,
                   RELATED_PRODUCTS,
                   BRAND]

# data pre-processing functions
TEXT_SUMMARY_FUNCTION = lambda x: len(x.split())


def _process_helpfulness(data):
    """
    Process helfulness field to obtain
    - # of Helpful Votes
    - # of Unhelpful Votes
    - Total # of Helpfulness Votes
    """
    num_helpful, total = ast.literal_eval(data)
    if num_helpful > total:
        return pd.Series([None, None, None],
                         index=[NUM_HELPFUL, NUM_UNHELPFUL, HELPFULNESS])
    elif total == 0:
        return pd.Series([0, 0, None],
                         index=[NUM_HELPFUL, NUM_UNHELPFUL, HELPFULNESS])
    else:
        return pd.Series([num_helpful, total - num_helpful, 1.0 * num_helpful / total],
                         index=[NUM_HELPFUL, NUM_UNHELPFUL, HELPFULNESS])


def _process_related_products(data):
    """
    Process related field (product table) to obtain
    - list of Product IDs bought together
    """
    try:
        # dict containing different lists of related products
        related_data_dict = ast.literal_eval(data)
    except:
        return None
    if BOUGHT_TOGETHER in related_data_dict:
        # products bought together from above dict
        return related_data_dict[BOUGHT_TOGETHER]
    return None


def _process_salesrank(data):
    """
    Process salesrank field (product table) to obtain
    - Product Category
    - Sales Rank
    """
    try:
        # data -> {'Books': 63}
        data_dict = ast.literal_eval(data)
    except:
        return pd.Series([None, None], index=[CATEGORY, SALES_RANK])

    # empty dict
    if data_dict == dict():
        return pd.Series([None, None], index=[CATEGORY, SALES_RANK])

    # ensure dict has length 1
    assert len(data_dict) == 1, str(data_dict)

    category = list(data_dict.keys())[0].replace('&amp;', '&')
    salesrank = data_dict[category]
    return pd.Series([category, salesrank], index=[CATEGORY, SALES_RANK])


def _load_data(category):
    """
    Load and process review data into pandas dataframe
    """
    filename = os.path.join(DATA_DIR, REVIEW_FILES[category])
    # used to obtain word counts
    converters = {REVIEW_TEXT: TEXT_SUMMARY_FUNCTION,
                  SUMMARY: TEXT_SUMMARY_FUNCTION
                  }

    review_df = pd.read_csv(filename,
                            converters=converters,
                            usecols=REVIEW_COLUMNS,
                            )
    review_df[[NUM_HELPFUL, NUM_UNHELPFUL, HELPFULNESS]] = review_df[HELPFULNESS].apply(_process_helpfulness)
    review_df = review_df.rename(columns={REVIEW_TEXT: REVIEW_WORD_COUNT, SUMMARY: SUMMARY_LENGTH})
    return review_df


def _load_metadata(category):
    """
    Load and process review metadat (or product data) into pandas dataframe
    """
    filename = os.path.join(DATA_DIR, METADATA_FILES[category])

    converters = {RELATED_PRODUCTS: _process_related_products,
                  PRODUCT_DESCRIPTION: TEXT_SUMMARY_FUNCTION}

    review_metadata_df = pd.read_csv(filename,
                                     converters=converters,
                                     usecols=PRODUCT_COLUMNS
                                     )
    review_metadata_df[[CATEGORY, SALES_RANK]] = review_metadata_df[SALES_RANK].apply(_process_salesrank)
    review_metadata_df = review_metadata_df.rename(columns={RELATED_PRODUCTS: BOUGHT_TOGETHER,
                                                            PRODUCT_DESCRIPTION: PRODUCT_DESCRIPTION_LENGTH})
    return review_metadata_df


def load_review_data(load_all_data=False):
    if load_all_data:
        review_df = []
        for category in REVIEW_FILES:
            review_df.append(_load_data(category))
        review_df = pd.concat(review_df)
        review_metadata_df = _load_metadata('all')
    else:
        review_df = _load_data('clothing')
        review_metadata_df = _load_metadata('clothing')
    return review_df, review_metadata_df
