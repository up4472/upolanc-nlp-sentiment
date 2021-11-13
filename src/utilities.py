from nltk.sentiment import SentimentIntensityAnalyzer

class NLTKAnalyzer :

	def __init__ (self) -> None :
		self.analyzer = SentimentIntensityAnalyzer()

	def get_polarity_score (self, text : str) -> dict :
		return self.analyzer.polarity_scores(text)

	def get_negative_score (self, text : str) -> float :
		return self.get_polarity_score(text)['neg']

	def get_neutral_score (self, text : str) -> float :
		return self.get_polarity_score(text)['neu']

	def get_positive_score (self, text : str) -> float :
		return self.get_polarity_score(text)['pos']

	def get_compound_score (self, text : str) -> float :
		return self.get_polarity_score(text)['compound']
