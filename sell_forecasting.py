import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint	
from tensorflow.keras.models import Sequential


## read and preparing datase
sale_dataset = pd.read_csv('sale_dataset.csv')
sale_dataset = sale_dataset.drop(['store','item'], axis=1)
sale_dataset['date'] = pd.to_datetime(sale_dataset['date'])
sale_dataset['date'] = sale_dataset['date'].dt.to_period('M')
month_sales = sale_dataset.groupby('date').sum().reset_index()
month_sales['date'] = month_sales['date'].dt.to_timestamp()

month_sales['sales_diff'] = month_sales['sales'].diff()
month_sales = month_sales.dropna()   
#print(month_sales)
supverised_sale_data = month_sales.drop(['date','sales'], axis=1)


# finding sell deferences and preparing supverised data
for i in range(1,13):
    column_name = 'month_' + str(i)
    supverised_sale_data[column_name] = supverised_sale_data['sales_diff'].shift(i)


supverised_sale_data = supverised_sale_data.dropna().reset_index(drop=True)  #remove the null value 
train_sale_data = supverised_sale_data[:-12]
test_sale_data = supverised_sale_data[-12:]
 
#print(train_data)   
scaler = MinMaxScaler(feature_range=(-1,1))
# Fitting the scaler on the training data and transforming the dat
scaler.fit(train_sale_data)
train_sale_data = scaler.transform(train_sale_data)
test_sale_data = scaler.transform(test_sale_data)

#print(train_data)   
#print(test_data)  

# Extracting features and labels for training data
X_train, y_train = train_sale_data[:,1:], train_sale_data[:,0:1]

# Extracting features and labels for testing data
X_test, y_test = test_sale_data[:,1:], test_sale_data[:,0:1]

# convert to 1-dimensional arrays
y_train = y_train.ravel()  
y_test = y_test.ravel() 
	
sales_dates = month_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
actual_sales = month_sales['sales'][-13:].to_list()


###########    XGBoost Regressor   #######
xgb_reg_model = XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror')
xgb_reg_model.fit(X_train, y_train)     # training finding relation between X_train and y_train
xgb_reg_pred = xgb_reg_model.predict(X_test)    #  predect X_test value
xgb_reg_pred = xgb_reg_pred.reshape(-1,1)
xgb_pred_test_set = np.concatenate([xgb_reg_pred,X_test], axis=1)    # combine predect with X_test
xgb_pred_test_set = scaler.inverse_transform(xgb_pred_test_set)

# Combine predictions with actual sales
re_list = []
for index in range(0, len(xgb_pred_test_set)):
    re_list.append(xgb_pred_test_set[index][0] + actual_sales[index])

# merge with the prediction DataFrame    
xgb_pred_series = pd.Series(re_list, name='xgb_pred')
predict_df = predict_df.merge(xgb_pred_series, left_index=True, right_index=True)

print(predict_df)
# Evaluate performance metrics
xgb_reg_rmse = np.sqrt(mean_squared_error(predict_df['xgb_pred'], month_sales['sales'][-12:]))
xgb_reg_mae = mean_absolute_error(predict_df['xgb_pred'], month_sales['sales'][-12:])
xgb_reg_r2 = r2_score(predict_df['xgb_pred'], month_sales['sales'][-12:])
print('XG Boost RMSE: ', xgb_reg_rmse)
print('XG Boost MAE: ', xgb_reg_mae)
print('XG Boost R2 Score: ', xgb_reg_r2)

# Showing predicted results in a graph
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['xgb_pred'],color="red")
plt.title("XG Boost Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()





########  Random Forest Regressor #########
rf_reg_model = RandomForestRegressor(n_estimators=100, max_depth=20)  
rf_reg_model.fit(X_train, y_train)    # training finding relation between X_train and y_train
rf_reg_pred = rf_reg_model.predict(X_test)   #  predect X_test value
rf_reg_pred = rf_reg_pred.reshape(-1,1)
rf_pred_test_set = np.concatenate([rf_reg_pred,X_test], axis=1)    # combine predect with X_test
rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)

re_list = []
for index in range(0, len(rf_pred_test_set)):
    re_list.append(rf_pred_test_set[index][0] + actual_sales[index])

# merge with the prediction DataFrame
rf_pred_series = pd.Series(re_list, name='rf_pred')
predict_df = predict_df.merge(rf_pred_series, left_index=True, right_index=True)

# Evaluate performance metrics
rf_reg_rmse = np.sqrt(mean_squared_error(predict_df['rf_pred'], month_sales['sales'][-12:]))
rf_reg_mae = mean_absolute_error(predict_df['rf_pred'], month_sales['sales'][-12:])
rf_reg_r2 = r2_score(predict_df['rf_pred'], month_sales['sales'][-12:])
print('Random Forest RMSE: ', rf_reg_rmse)
print('Random Forest MAE: ', rf_reg_mae)
print('Random Forest R2 Score: ', rf_reg_r2)

# Showing predicted results in a graph
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['rf_pred'],color="orange")
plt.title("Random Forest Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()



########  Linear Regression  ################

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)    # training finding relation between X_train and y_train
linear_reg_pred = linear_reg_model.predict(X_test)  #  predect X_test value
linear_reg_pred = linear_reg_pred.reshape(-1,1)
linreg_pred_test_set = np.concatenate([linear_reg_pred,X_test], axis=1)   # combine predect with X_test
linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set) 

# Combine predictions with actual sales
re_list = []
for index in range(0, len(linreg_pred_test_set)):
    re_list.append(linreg_pred_test_set[index][0] + actual_sales[index])

# merge with the prediction DataFrame
linreg_pred_series = pd.Series(re_list,name='linreg_pred')
predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)

# Evaluate performance metrics
linear_reg_rmse = np.sqrt(mean_squared_error(predict_df['linreg_pred'], month_sales['sales'][-12:]))
linreg_mae = mean_absolute_error(predict_df['linreg_pred'], month_sales['sales'][-12:])
linreg_r2 = r2_score(predict_df['linreg_pred'], month_sales['sales'][-12:])
print('Linear Regression RMSE: ', linear_reg_rmse)
print('Linear Regression MAE: ', linreg_mae)
print('Linear Regression R2 Score: ', linreg_r2)

# Showing predicted results in a graph
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['linreg_pred'],color="brown")
plt.title("Linear Regression Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()

####################




########    Ridge Regression    ############

ridge_model = Ridge(alpha=1.0)  # You can experiment with different values of alpha
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

# Inverse transform to get predictions in original scale
ridge_pred = ridge_pred.reshape(-1, 1)
ridge_pred_test_set = np.concatenate([ridge_pred, X_test], axis=1)
ridge_pred_test_set = scaler.inverse_transform(ridge_pred_test_set)

# Combine predictions with actual sales
result_list_ridge = []       
for index in range(0, len(ridge_pred_test_set)):
    result_list_ridge.append(ridge_pred_test_set[index][0] + actual_sales[index])

# merge with the prediction DataFrame
ridge_pred_series = pd.Series(result_list_ridge, name='ridge_pred')
predict_df = predict_df.merge(ridge_pred_series, left_index=True, right_index=True)

# Evaluate performance metrics
ridge_rmse = np.sqrt(mean_squared_error(predict_df['ridge_pred'], month_sales['sales'][-12:]))
ridge_mae = mean_absolute_error(predict_df['ridge_pred'], month_sales['sales'][-12:])
ridge_r2 = r2_score(predict_df['ridge_pred'], month_sales['sales'][-12:])
print('Ridge  Regression RMSE: ', ridge_rmse)
print('Ridge  Regression MAE: ', ridge_mae)
print('Ridge  Regression R2 Score: ', ridge_r2)

# Showing predicted results in a graph
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['ridge_pred'],color="blue")
plt.title("Ridge Regression Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()



################   using LSTM RNN
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Creating a Sequential model
model = Sequential()
#Adding an LSTM layer with 4 units,
model.add(LSTM(4, batch_input_shape=(1, X_train_lstm.shape[1], X_test_lstm.shape[2])))
# Adding a Dense layer with 10 units and ReLU activation function
model.add(Dense(10, activation='relu'))
# Adding a Dense layer with 1 unit (output layer)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Creating a list of callbacks including EarlyStopping and the ModelCheckpoint
checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
callbacks = [EarlyStopping(patience=5), model_checkpoint_callback]

#Training the model
history = model.fit(X_train_lstm, y_train, epochs=200, batch_size=1, validation_data=(X_test_lstm, y_test), callbacks=callbacks)

metrics_df = pd.DataFrame(history.history)
print(metrics_df)



lstm_pred = model.predict(X_test_lstm, batch_size=1)
lstm_pred = lstm_pred.reshape(-1,1)
lstm_pred_test_set = np.concatenate([lstm_pred,X_test], axis=1)
lstm_pred_test_set = scaler.inverse_transform(lstm_pred_test_set)
re_list = []
for index in range(0, len(lstm_pred_test_set)):
    re_list.append(lstm_pred_test_set[index][0] + actual_sales[index])
lstm_pred_series = pd.Series(re_list, name='lstm_pred')
predict_df = predict_df.merge(lstm_pred_series, left_index=True, right_index=True)


lstm_rmse = np.sqrt(mean_squared_error(predict_df['lstm_pred'], month_sales['sales'][-12:]))
lstm_mae = mean_absolute_error(predict_df['lstm_pred'], month_sales['sales'][-12:])
lstm_r2 = r2_score(predict_df['lstm_pred'], month_sales['sales'][-12:])
print('LSTM RMSE: ', lstm_rmse)
print('LSTM MAE: ', lstm_mae)
print('LSTM R2 Score: ', lstm_r2)

plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(month_sales['date'], month_sales['sales'],color="green")
plt.plot(predict_df['date'], predict_df['lstm_pred'],color="black")
plt.title("LSTM Predict")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()

# Compare the performance of all the models
xgb_stats = [xgb_reg_rmse, xgb_reg_mae, xgb_reg_r2]
rf_stats = [rf_reg_rmse, rf_reg_mae, rf_reg_r2]
linreg_stats = [linear_reg_rmse, linreg_mae, linreg_r2]
rg_stats = [ridge_rmse, ridge_mae, ridge_r2]
lstm_stats = [lstm_rmse, lstm_mae, lstm_r2]

plt.figure(figsize=(15,7))
plt.plot(linreg_stats)
plt.plot(rf_stats)
plt.plot(xgb_stats)
plt.plot(rg_stats)
plt.plot(lstm_stats)
plt.title("Model Comparison between Linear Regression, Random Forest, XG Boost,Ridge Regression and LSTM")
plt.xticks([0,1,2], labels=['RMSE','MAE','R2 Score'])
plt.legend(["Linear Regression", "Random Forest", "XG Boost", "Ridge Regression","LSTM"])
plt.show()
