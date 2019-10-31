# gradient-descent-for-linear-regression
Hand written implementation of gradient descent for linear regression that uses the "kc_house_data" from Kaggle for model training and testing.

The KC House dataset contains feature information describing the characteristics of various apartment properties, and a response column detailing the respective property's price.  Conveniently, the training and testing data were already split into training and testing CSV files.

The enclosed python file first reads in the training and testing data before pre-processing and standardization.  It then runs the data through a gradient-descent-for-linear-regression algorithm that trains a model and attempts to make accurate predictions.  
