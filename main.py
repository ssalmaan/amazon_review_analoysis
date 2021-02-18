import argparse
import seaborn as sns
import random
from tabulate import tabulate

from load_data import *
from analysis import *

# Q2
def review_behavior_by_category(joined_df, metric, num_top_categories=10):
    """
    Group reviews by category and
    - obtain # of reviews per # of products in each category
    - select top 10 categories using this criteria
    - sort data by average metric, e.g. RATING, REVIEW_WORD_COUNT etc.
    """
    grouped_df = joined_df[[CATEGORY, metric, PRODUCT_ID]].groupby(CATEGORY) \
        .agg({metric: ['mean', 'count'], PRODUCT_ID: ['nunique']})
    # flatten 2-level column hierarchy
    grouped_df.columns = ["_".join(x) for x in grouped_df.columns.ravel()]

    # reviews_per_product -> # of reviews / # of products
    grouped_df['reviews_per_product'] = grouped_df[f'{metric}_count'] / grouped_df[f'{PRODUCT_ID}_nunique']
    grouped_df = grouped_df.nlargest(num_top_categories, f'{metric}_count').sort_values(f'{metric}_mean',
                                                                                        ascending=False)
    return grouped_df.reset_index()


# Q3
def get_average_metric_with_price(data, metadata, metric):
    """
    Group data per product
    - product price
    - average metric, e.g. RATING, REVIEW_WORD_COUNT etc.
    """
    # aggregate metric (e.g. rating) in review table
    product_metric_df = data[[PRODUCT_ID, metric]].groupby(PRODUCT_ID).agg({metric: ['mean']})
    product_metric_df.columns = ["_".join(x) for x in product_metric_df.columns.ravel()]

    # join with product table to get price and category
    product_price_metric_df = product_metric_df \
        .join(metadata[[PRODUCT_ID, PRICE, CATEGORY]] \
              .set_index(PRODUCT_ID)) \
        .reset_index()

    # drop products with price missing if any
    product_price_metric_df = product_price_metric_df[product_price_metric_df[PRICE].notna()]
    return product_price_metric_df


# Q5
def sample_products_bought_together(data, metadata, num_samples=10000):
    """
    - Select random product : P1
    - Select random product bought together with it : P2
    - Obtain Average rating of P1 : R1
    - Obtain Average rating of P2 : R2
    - Add [P1, R1, P2, R2] to a dataframe
    Repeat the above steps until we have 10000 product pairs
    """
    product_rating_df = data[[PRODUCT_ID, RATING]].groupby(PRODUCT_ID).agg({RATING: ['mean']})
    product_rating_df.columns = ["_".join(x) for x in product_rating_df.columns.ravel()]

    bought_together_df = metadata[[PRODUCT_ID, BOUGHT_TOGETHER]]
    bought_together_df = bought_together_df[bought_together_df[BOUGHT_TOGETHER].notnull()]
    rating_pair_df = pd.DataFrame(columns=['Product_ID_1', 'Rating_1', 'Product_ID_2', 'Rating_2'])

    found_samples = 0
    while found_samples < num_samples:
        random_record = bought_together_df.sample().iloc[0]
        item1 = random_record[PRODUCT_ID]
        item2 = random.choice(ast.literal_eval(random_record[BOUGHT_TOGETHER]))

        try:
            rating1 = product_rating_df._get_value(item1, f'{RATING}_mean')
            rating2 = product_rating_df._get_value(item2, f'{RATING}_mean')
        except KeyError:
            continue

        rating_pair_df = rating_pair_df.append({'Product_ID_1': item1,
                                                'Rating_1': rating1,
                                                'Product_ID_2': item2,
                                                'Rating_2': rating2},
                                               ignore_index=True)
        found_samples += 1

    return rating_pair_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, nargs='?', default='data',
                        help='Data folder path')
    parser.add_argument('--result_dir', type=str, nargs='?', default='results',
                        help='Result folder path')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='Directly load pre-processed data')
    parser.add_argument('--no-preload', dest='preload', action='store_false',
                        help='Pre-process raw data')
    parser.set_defaults(preload=True)
    parser.add_argument('--analysis_type', type=int, nargs='?',
                        default=1,
                        help='Question number?')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RESULT_DIR = args.result_dir

    analysis_type = args.analysis_type
    assert analysis_type in [1, 2, 3, 4, 5, 6]

    if args.preload:
        # load dataframes from existing CSVs
        review_df = pd.read_csv('data/review_df.csv')
        review_metadata_df = pd.read_csv('data/review_metadata_df.csv')
    else:
        # load and process data and save to CSVs
        review_df, review_metadata_df = load_review_data(load_all_data=False)
        review_df.to_csv('data/review_df.csv')
        review_metadata_df.to_csv('data/review_metadata_df.csv')

    joined_df = review_df.set_index(PRODUCT_ID).join(review_metadata_df.set_index(PRODUCT_ID),
                                                     lsuffix='', rsuffix='_right').reset_index()

    if analysis_type == 1:
        # Q1: What is the relation between the reviews and the helpfulness?
        plot_histogram(f'{RESULT_DIR}/Q1/helpfulness_histogram.png', review_df, HELPFULNESS)
        plot_histogram(f'{RESULT_DIR}/Q1/num_helpful_histogram.png', review_df, NUM_HELPFUL, log=True,
                       max_samples=10000)
        plot_histogram(f'{RESULT_DIR}/Q1/num_unhelpful_histogram.png', review_df, NUM_UNHELPFUL, log=True,
                       max_samples=10000, bins=100)

        joint_plot(f'{RESULT_DIR}/Q1/wordcount_helpfulness_joint.png', review_df, REVIEW_WORD_COUNT, HELPFULNESS,
                   alpha=0.25, max_samples=10000)
        joint_plot(f'{RESULT_DIR}/Q1/rating_helpfulness_joint.png', review_df, RATING, HELPFULNESS,
                   max_samples=10000, alpha=0.25)

        print_correlation(review_df, HELPFULNESS, RATING, "helpfulness", "rating")
        print_correlation(review_df, HELPFULNESS, REVIEW_WORD_COUNT, "helpfulness", "review word count")
        print_correlation(review_df, HELPFULNESS, SUMMARY_LENGTH, "helpfulness", "summary length")

        sns.barplot(x=RATING, y=HELPFULNESS, data=review_df, ci='sd', capsize=.2)
        plt.xlabel(get_label(RATING))
        plt.ylabel(get_label(HELPFULNESS))
        plt.savefig(f'{RESULT_DIR}/Q1/rating_helpfulness_barplot.png', bbox_inches='tight')
        plt.close()

        joint_plot(f'{RESULT_DIR}/Q1/helpfulness_rating_jointplot.png', review_df, HELPFULNESS, RATING,
                   max_samples=10000, alpha=0.25)

    elif analysis_type == 2:
        # Q2: What is the review behavior among different categories?
        metric = RATING
        review_by_category_df = review_behavior_by_category(joined_df, metric)
        g = sns.scatterplot(f'{metric}_mean',
                        'reviews_per_product',
                        data=review_by_category_df,
                        hue=CATEGORY)
        plt.xlabel(get_label(f'{metric}_mean'))
        plt.ylabel(get_label('Reviews per Product'))
        plt.legend(loc='upper left')
        plt.savefig(f'{RESULT_DIR}/Q2/rating_numreviews_scatterplot.png', bbox_inches='tight')
        plt.close()


        print(tabulate(review_by_category_df, headers='keys', tablefmt='psql'))

        top_categories = review_by_category_df[CATEGORY].tolist()
        filtered_df = joined_df[joined_df[CATEGORY].isin(top_categories)]
        sns.barplot(CATEGORY, metric, data=filtered_df, ci='sd', capsize=.2)
        plt.xlabel(get_label(CATEGORY))
        plt.ylabel(get_label(metric))
        plt.xticks(rotation=90)
        plt.savefig(f'{RESULT_DIR}/Q2/category_rating_barplot.png', bbox_inches='tight')
        plt.close()

    elif analysis_type == 3:
        # Q3: Is there a relationship between price and reviews?

        # correlation with feature aggregation per-product, e.g. average rating per product
        product_price_wordcount_df = get_average_metric_with_price(review_df, review_metadata_df, REVIEW_WORD_COUNT)
        print_correlation(product_price_wordcount_df,
                          PRICE, f'{REVIEW_WORD_COUNT}_mean',
                          'Price', 'Mean Review Word Count')
        g = sns.regplot(data=product_price_wordcount_df.sample(250), x=PRICE, y=f'{REVIEW_WORD_COUNT}_mean', marker='+',
                    scatter_kws={'alpha': 0.4})
        plt.xlabel(get_label(PRICE))
        plt.ylabel(get_label(f'{REVIEW_WORD_COUNT}_mean'))
        plt.savefig(f'{RESULT_DIR}/Q3/price_wordcount_scatterplot.png', bbox_inches='tight')
        plt.close()

        product_price_rating_df = get_average_metric_with_price(review_df, review_metadata_df, RATING)
        print_correlation(product_price_rating_df,
                          PRICE, f'{RATING}_mean',
                          'Price', 'Mean Rating')
        g = sns.regplot(data=product_price_rating_df.sample(250), x=PRICE, y=f'{RATING}_mean', marker='+',
                    scatter_kws={'alpha': 0.4})
        plt.xlabel(get_label(PRICE))
        plt.ylabel(get_label(f'{RATING}_mean'))
        plt.savefig(f'{RESULT_DIR}/Q3/price_rating_scatterplot.png', bbox_inches='tight')
        plt.close()

        # product category wise analysis
        review_by_category_df = review_behavior_by_category(joined_df, REVIEW_WORD_COUNT)
        top_categories = review_by_category_df[CATEGORY].tolist()

        # price_wordcount_facetplot
        filtered_df = product_price_wordcount_df[product_price_wordcount_df[CATEGORY].isin(top_categories)]
        g = sns.FacetGrid(filtered_df, col=CATEGORY, height=5, col_wrap=2)
        g.map(sns.scatterplot, PRICE, f'{REVIEW_WORD_COUNT}_mean', marker='+', alpha=0.3)
        g.set_axis_labels(get_label(PRICE), get_label(f'{REVIEW_WORD_COUNT}_mean'))
        plt.xlim(0, 300)
        plt.ylim(0, 300)
        plt.savefig(f'{RESULT_DIR}/Q3/price_wordcount_facetplot.png', bbox_inches='tight')
        plt.close()

        # price_rating_facetplot
        filtered_df = product_price_rating_df[product_price_rating_df[CATEGORY].isin(top_categories)]
        g = sns.FacetGrid(filtered_df, col=CATEGORY, height=5, col_wrap=2)
        g.map(sns.regplot, PRICE, f'{RATING}_mean', marker='+', scatter_kws={'alpha': 0.2})
        # plt.xlabel(get_label(PRICE))
        # plt.ylabel(get_label(f'{RATING}_mean'))
        g.set_axis_labels(get_label(PRICE), get_label(f'{RATING}_mean'))
        plt.xlim(0, 300)
        plt.ylim(2, 5)
        plt.savefig(f'{RESULT_DIR}/Q3/price_rating_facetplot.png', bbox_inches='tight')
        plt.close()

    elif analysis_type == 4:
        reviewer_summary_df = joined_df.groupby([REVIEWER_ID, REVIEWER_NAME]) \
            .agg({PRICE: ['mean'], PRODUCT_ID: ['count']})

        reviewer_summary_df.columns = ["_".join(x) for x in reviewer_summary_df.columns.ravel()]
        reviewer_summary_df['Number of Reviews'] = reviewer_summary_df[f'{PRODUCT_ID}_count']
        joint_plot(f'{RESULT_DIR}/Q4/price_numreviews_jointplot.png', reviewer_summary_df,
                   f'{PRICE}_mean', 'Number of Reviews', max_samples=500, alpha=0.6)

    elif analysis_type == 5:
        rating_pair_df = sample_products_bought_together(review_df, review_metadata_df, num_samples=1000)
        print(tabulate(rating_pair_df.head(10), headers='keys', tablefmt='psql'))

        print_correlation(rating_pair_df, 'Rating_1', 'Rating_2', 'Rating_1', 'Rating_2')

        joint_plot(f'{RESULT_DIR}/Q5/products_bought_together_jointplot.png', rating_pair_df, 'Rating_1', 'Rating_2', max_samples=1000, alpha=0.25)

    elif analysis_type == 6:
        YEAR = 'year'
        time_joined_df = review_df.set_index(PRODUCT_ID).join(review_metadata_df.set_index(PRODUCT_ID),
                                                              lsuffix='', rsuffix='_right').reset_index()

        metric = REVIEW_WORD_COUNT
        rating_df = time_joined_df[[YEAR, PRODUCT_ID, metric]].groupby([YEAR, PRODUCT_ID]).mean(metric).reset_index()
        joint_plot(f'{RESULT_DIR}/Q6/year_wordcount_jointplot.png', rating_df, YEAR, metric, alpha=0.75, max_samples=10000)
        plt.ylim(0, 600)
        plt.close()

        metric = RATING
        rating_df = time_joined_df[[YEAR, PRODUCT_ID, metric]].groupby([YEAR, PRODUCT_ID]).mean(metric).reset_index()
        joint_plot(f'{RESULT_DIR}/Q6/year_rating_jointplot.png', rating_df, YEAR, metric, alpha=0.75, max_samples=10000)
        plt.close()

        metric = HELPFULNESS
        rating_df = time_joined_df[[YEAR, PRODUCT_ID, metric]].groupby([YEAR, PRODUCT_ID]).mean(metric).reset_index()
        joint_plot(f'{RESULT_DIR}/Q6/year_helpfulness_jointplot.png', rating_df, YEAR, metric, alpha=0.75, max_samples=10000)
        plt.close()