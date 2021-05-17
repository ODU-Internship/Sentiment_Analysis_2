# Sentiment_Analysis_2

routes:
* /predict = pass json data with key='emails', value='list of text to be predicted on'
ex: {'emails':['msg1','msg2']}

* '/retrain' = call to invoke retrain function.
	2 kinds of retrain available:
	* based on predicted data whose labels are corrected by supervisor, runs only when more than 10000 such msg collected, and UTC time is 12am - 3am. A post request needs to be made at this time to invoke and execute.
	* using train data uploaded by supervisor - here once supervisor clicks retrain button, api call generated

	returns string consisting of features (accuracy, score, etc) of newly trained model, which can be displayed to supervisor to check model quality.

* '/replace' = call to use newly trained model in production
	pass json data with key="merge" and value = "yes" (to deploy new model)
  
  NOTE: The model in models folder needs to be unzipped before invoking API calls
