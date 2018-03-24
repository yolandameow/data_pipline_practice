import luigi
import pandas as pd
import numpy as np
import pickle as pkl
from ast import literal_eval
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """

    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def run(self):
        tweets = pd.read_csv(self.tweet_file, skiprows=0, low_memory=False)
        tweets = tweets[pd.notnull(tweets['tweet_coord'])]
        tweets = tweets[tweets.tweet_coord != '[0.0, 0.0]']
        tweets.to_csv(self.output_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def run(self):
        cities = pd.read_csv(self.cities_file, skiprows=0, low_memory=False)

        with self.input().open('r') as input_data:
            clean_data = pd.read_csv(input_data)

            airline_sentiment = clean_data['airline_sentiment']
            airline_sentiment = airline_sentiment.map({"negative": "0", "neutral": "1", "positive": "2"})

            tweet_coord = clean_data['tweet_coord']
            city_coord = cities[['name', 'latitude', 'longitude']]

            tweet_city_name = self.closestCity(tweet_coord, city_coord)
            print "\n-------\n city name", tweet_city_name
            one_hot_cities = pd.get_dummies(tweet_city_name['city_name'].astype('category', categories=np.sort(city_coord['name'].unique())))
            output_features = pd.concat([airline_sentiment, tweet_city_name], axis=1)
            output_features = output_features.join(one_hot_cities)
            output_features.to_csv(self.output_file)

    ## The method to compute the distance between a datapoint and a city
    def distance(self, datapoint, city):
        return np.linalg.norm(datapoint - city)

    ## The method to get the cloest city DataFrame
    def closestCity(self, tweet_coords, city_coords):
        tweet_location = pd.DataFrame(columns=['city_name'])

        for i in range(len(tweet_coords)):
            print 'Getting location for tweet #', i
            distances = pd.DataFrame(columns=['distance'])
            tweet_coord = tweet_coords[i]
            tweet_coord = tweet_coord[1:-1].split(',')

            theta = 5

            filter_city_coords = (((float(tweet_coord[0])) - theta) <= city_coords['latitude']) & (
                    ((float(tweet_coord[0])) + theta) >= city_coords['latitude']) & (
                                         ((float(tweet_coord[1])) - theta) <= city_coords['longitude']) & (
                                         ((float(tweet_coord[1])) + theta) >= city_coords['longitude'])

            city_coords_filtered = city_coords

            city_coords_filtered = city_coords_filtered[filter_city_coords]

            city_coords_filtered = city_coords_filtered.reset_index(drop=True)

            for j in range(len(city_coords_filtered)):
                city_lat = city_coords_filtered['latitude'][j]
            city_lon = city_coords_filtered['longitude'][j]
            tweet_coord = [float(k) for k in tweet_coord]
            dist = self.distance(np.asarray(tweet_coord), np.asarray([city_lat, city_lon]))
            distances.loc[j] = dist

            tweet_location.loc[i] = city_coords_filtered['name'][distances['distance'].idxmin()]

        return tweet_location

    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file)

    def run(self):
        with self.input().open('r') as input_data:
            features = pd.read_csv(input_data, skiprows=0, low_memory=False)
            train_y = features.iloc[:,1]
            #print "train_y:",train_y
            train_x = features.iloc[:,3:]

            ###=====Logistic Regression Model=====#####
            model = LogisticRegression()
            model.fit(train_x, train_y)

            ###=====Decision Tree Model=====#####
            #model = DecisionTreeClassifier()
            #model.fit(train_x, train_y)

            cities = list(train_x)

            with self.output().open('w') as m:
                pkl.dump([model, cities], m)


    def output(self):
        return luigi.LocalTarget(self.output_file)


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    def requires(self):
        return TrainModelTask(self.tweet_file)

    def run(self):
        print ("\n\n\n\n Method ScoreTask")
        with self.input().open('r') as input_data:
            model, cities = pkl.load(input_data)
            city_code = pd.DataFrame(np.identity(len(cities)), columns=cities)
            sentiment_prob = model.predict_proba(city_code)

            pred_scores = pd.DataFrame(cities, columns=['city_name'])
            pred_scores[['negative_probability', 'neutral_probability', 'positive_probability']] = pd.DataFrame(sentiment_prob, dtype=float)
            pred_scores.sort_values(['positive_probability'], ascending=False, inplace=True)

            pred_scores.to_csv(self.output_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)


if __name__ == "__main__":
    luigi.run()
