import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
from sklearn.model_selection import train_test_split

st.title('Prediksi Harga Emas Menggunakan Metode Linear Regression')

# Mendapatkan data harga emas dari URL
df_data = pd.read_csv('https://raw.githubusercontent.com/FajarAndrianto037/kelompokpro/main/PLN%3DX.csv')
df_data.head(7)
#df.isnull().sum()
#df['Open'] = df['Open'].fillna(value=df['Open'].median())
#df['High'] = df['High'].fillna(value=df['High'].median())
#df['Low'] = df['Low'].fillna(value=df['Low'].median())
#df['Close'] = df['Close'].fillna(value=df['Close'].median())
#df['Adj Close'] = df['Adj Close'].fillna(value=df['Adj Close'].median())
#df['Volume'] = df['Volume'].fillna(value=df['Volume'].median())


# Konversi kolom 'Date' menjadi tipe datetime
#df['Date'] = pd.to_datetime(df['Open'])

# Menampilkan data harga emas
st.write('Data:', df_data)

# Memisahkan data menjadi data train dan data test
data = df_data['Close']
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)

# Praproses data dengan MinMaxScaler
scaler= MinMaxScaler()
X_norm= scaler.fit_transform(df_X)
# y_norm= scaler.fit_transform(df_y)

# reshaped_data = data.reshape(-1, 1)
#train = pd.DataFrame(train_scaled, columns = ['data'])
#train = train['data']
# st.write('Data Train',train)

#test = pd.DataFrame(test_scaled, columns = ['data'])
#test = test['data']

# Menggabungkan train dan test menjadi satu tabel
#merged_data = pd.concat([train, test], axis=1)
#merged_data.columns = ['Train', 'Test']
# Menampilkan tabel hasil penggabungan
st.write('Hasil Normalisasi:')
st.write(merged_data)

# Membentuk sequence untuk training dan testing
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X = array(X)
    y = array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Perubahan bentuk data
    return X, y

# Menyiapkan data train
#X_train, Y_train = split_sequence(train_scaled, 2)
#x_train = pd.DataFrame(X_train.reshape((X_train.shape[0], X_train.shape[1])), columns=['xt-2', 'xt-1'])  # Perubahan bentuk data
#y_train = pd.DataFrame(Y_train, columns=['xt'])
#dataset_train = pd.concat([x_train, y_train], axis=1)
#X_train = dataset_train.iloc[:, :2].values
#Y_train = dataset_train.iloc[:, -1].values

# Menyiapkan data test
#test_x, test_y = split_sequence(test_scaled, 2)
#x_test = pd.DataFrame(test_x.reshape((test_x.shape[0], test_x.shape[1])), columns=['xt-2', 'xt-1'])  # Perubahan bentuk data
#y_test = pd.DataFrame(test_y, columns=['xt'])
#dataset_test = pd.concat([x_test, y_test], axis=1)
# st.write('Dataset Test:', dataset_test)
#X_test = dataset_test.iloc[:, :2].values
#Y_test = dataset_test.iloc[:, -1].values

# Melakukan prediksi dengan Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
knn_preds = regressor.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, Y_train)
dt_preds = dt_model.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, Y_train)
rf_preds = rf_model.predict(X_test)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
lr_preds = lr_model.predict(X_test)

from sklearn.svm import SVR
svr_model = SVR()
svr_model.fit(X_train, Y_train)
svr_preds = svr_model.predict(X_test)

# import knn
from sklearn.neighbors import KNeighborsRegressor
model_knn = KNeighborsRegressor(n_neighbors=7)
model_knn.fit(X_train, y_train)
y_pred=model_knn.predict(X_test)

#naivebayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# Create a Naive Bayes
naive_bayes = GaussianNB()
import numpy as np

# Define the bin edges or thresholds
bin_edges = [4.0, 4.5, 5.0]  # Adjust the values based on your requirements

# Perform binning on the labels
y_train_categorical = np.digitize(y_train, bin_edges)

# Create a Naive Bayes classifier
naive_bayes = GaussianNB()

# Training the model
naive_bayes.fit(X_train, y_train_categorical)

# Making predictions on the test set
y_pred = naive_bayes.predict(X_test)

#RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Create a Random Forest Regressor
random_forest = RandomForestRegressor()
# Train the model
random_forest.fit(X_train, y_train)

# Make predictions
predictions = random_forest.predict(X_test)

# Mengembalikan skala data ke aslinya
model_knn = scaler.inverse_transform(model_knn.reshape(-1, 1))
naive_bayes = scaler.inverse_transform(naive_bayes.reshape(-1, 1))
random_forest = scaler.inverse_transform(random_forest.reshape(-1, 1))
#lr_preds = scaler.inverse_transform(lr_preds.reshape(-1, 1))
#svr_preds = scaler.inverse_transform(svr_preds.reshape(-1, 1))

reshaped_datates = Y_test.reshape(-1, 1)
actual_test = scaler.inverse_transform(reshaped_datates)

# Menyimpan hasil prediksi dan data aktual dalam file Excel
prediksi_knn = pd.DataFrame(model_knn)
prediksi_nb = pd.DataFrame(naive_bayes)
prediksi_rf = pd.DataFrame(random_forest)
#prediksi_lr = pd.DataFrame(lr_preds)
#prediksi_svr = pd.DataFrame(svr_preds)

actual = pd.DataFrame(actual_test)

# Menghitung mean absolute percentage error (MAPE)
knn_mape = mean_absolute_percentage_error(model_knn, actual_test) * 100
nb_mape = mean_absolute_percentage_error(naivebayes, actual_test) * 100
rf_mape = mean_absolute_percentage_error(random_forest, actual_test)* 100
#lr_mape = mean_absolute_percentage_error(lr_preds, actual_test) * 100
#svr_mape = mean_absolute_percentage_error(svr_preds, actual_test) * 100

# Menampilkan hasil prediksi dan MAPE
# st.write("Hasil Prediksi:")
# st.write(prediksi_knn)
# st.write("Data Aktual:")
# st.write(aktual)
st.write("MAPE KNeighborsRegressor:", knn_mape)
st.write("MAPE NaiveBayesRegressor:", nb_mape)
st.write("MAPE RandomForestRegressor:", rf_mape)


# Input tanggal untuk memprediksi harga emas
st.sidebar.title("Prediksi Harga Saham")
selected_date = st.sidebar.date_input("Pilih Tanggal")

if selected_date is not None:
    # Ubah tanggal menjadi format yang sesuai dengan data
    selected_date_str = selected_date.strftime("%Y-%m-%d")

    # Cari indeks tanggal terdekat dalam data
    closest_date = pd.Timestamp(selected_date)  # Convert to pd.Timestamp
    closest_date_idx = df['Date'].sub(closest_date).abs().idxmin()

    # Ambil data sebelum dan pada tanggal yang dipilih
    selected_data = df.loc[closest_date_idx-2:closest_date_idx, 'Open']

    # Praproses data dengan MinMaxScaler
    selected_data_scaled = scaler.transform(selected_data.values.reshape(-1, 1))

    # Bentuk input sequence untuk prediksi
    X_selected, _ = split_sequence(selected_data_scaled, 2)
    x_selected = pd.DataFrame(X_selected.reshape((X_selected.shape[0], X_selected.shape[1])), columns=['xt-2', 'xt-1'])
    X_selected = x_selected.values

    # Prediksi harga emas pada tanggal yang dipilih
    predicted_price_scaled = regressor.predict(X_selected)
    predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))

    st.sidebar.write("Prediksi Harga Saham pada Tanggal", selected_date_str)
    st.sidebar.write(predicted_price)