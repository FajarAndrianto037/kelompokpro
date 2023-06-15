
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import altair as alt
import pickle


Data,Preproses,Modelling,Implementasi = st.tabs(['Data','Preprosessing Data','Modelling','Implementasi'])

with Data:
   st.title("""
   Peramalan Data Time Series Pada Saham PLN.
   """)
   st.write('Proyek Sain Data')
   st.text("""
            1. Fajar Andrianto 200411100037 
            2. Rania Nuraini 200411100168   
            """)
   st.subheader('Tentang Dataset')
   st.write ("""
   Dataset yang digunakan adalah data time series pada Saham PLN, datanya di dapatkan dari website pada link berikut ini.
   """)
   st.write ("""
    Dataset yang digunakan berjumlah 262 data dan terdapat 7 atribut : 
    """)
   st.write('1. Date : berisi tanggal jalannya perdagangan mulai dari tanggal 15 juni 2022- 15 juni 2023')
   st.write('2. Open : berisi Harga pembukaan pada hari tersebut')
   st.write('3. High : berisi Harga tertinggi pada hari tersebut')
   st.write('4. Low : berisi Harga terendah pada hari tersebut')
   st.write('5. Close : berisi Harga penutup pada hari tersebut')
   st.write('6. Adj. Close : berisi Harga penutupan yang disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverset')
   st.write('7. Volume : berisi Volume perdagangan (dalam satuan lembar)')
   st.subheader('Dataset')
   df_data = pd.read_csv('https://raw.githubusercontent.com/FajarAndrianto037/kelompokpro/main/PLN%3DX.csv')
   df_data
   st.write('Dilakukan Pengecekan data kosong (Missing Value)')
   st.write(df_data.isnull().sum())
   st.write('Masih Terdapat data kosong maka dilakukan penanganan dengan mengisinya dengan nilai median')
   df_data['Open'] = df_data['Open'].fillna(value=df_data['Open'].median())
   df_data['High'] = df_data['High'].fillna(value=df_data['High'].median())
   df_data['Low'] = df_data['Low'].fillna(value=df_data['Low'].median())
   df_data['Close'] = df_data['Close'].fillna(value=df_data['Close'].median())
   df_data['Adj Close'] = df_data['Adj Close'].fillna(value=df_data['Adj Close'].median())
   df_data['Volume'] = df_data['Volume'].fillna(value=df_data['Volume'].median())
   st.write('Setelah dilakukan penanganan')
   st.write(dfdf_data.isnull().sum())
   st.write('Data yang akan di gunakan adalah data Close')


with Preproses:
   # untuk mengambil data yang akan diproses
   df_close= df_data['Close']
   # menghitung jumlah data
   n = len(data)
   # membagi data menjadi 80% untuk data training dan 20% data testing
   X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)
   st.write("""Dilakukan split data menjadi 80% data training dan 20% data testing""")
   st.write("""Dilakukan Normalisasi Menggunakan MinMax Scaler""")
   min_ = st.checkbox('MinMax Scaler')
   mod = st.button("Cek")
   # melakukan normalisasi menggunakan minMaxScaler
   from sklearn.preprocessing import MinMaxScaler
   scaler= MinMaxScaler()

# y_norm= scaler.fit_transform(df_y)
   # Mengaplikasikan MinMaxScaler pada data pengujian
   X_norm= scaler.fit_transform(df_X)
   # reshaped_data = data.reshape(-1, 1)
   X_norm= scaler.fit_transform(df_X)
   if min_:
      if mod:
         st.write("Data Training MinMax Scaler")
         train
         st.write("Data Test MinMax Scaler")
         train

   # transform univariate time series to supervised learning problem
   from numpy import array
    # split a univariate sequence into samples
   def split_sequence(sequence, n_steps):
      X, y = list(), list()
      for i in range(len(sequence)):
      # find the end of this pattern
         end_ix = i + n_steps
      # check if we are beyond the sequence
         if end_ix > len(sequence)-1:
               break
      # gather input and output parts of the pattern
         # print(i, end_ix)
         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
         X.append(seq_x)
         y.append(seq_y)
      return array(X), array(y)
   #memanggil fungsi untuk data training
   df_X, df_Y = split_sequence(train, 4)
   x = pd.DataFrame(df_X, columns = ['xt-4','xt-3','xt-2','xt-1'])
   y = pd.DataFrame(df_Y, columns = ['xt'])
   dataset_train = pd.concat([x, y], axis=1)
   dataset_train.to_csv('data-train.csv', index=False)
   X_train = dataset_train.iloc[:, :4].values
   Y_train = dataset_train.iloc[:, -1].values
   #memanggil fungsi untuk data testing
   test_x, test_y = split_sequence(test, 4)
   x = pd.DataFrame(test_x, columns = ['xt-4','xt-3','xt-2','xt-1'])
   y = pd.DataFrame(test_y, columns = ['xt'])
   dataset_test = pd.concat([x, y], axis=1)
   dataset_test.to_csv('data-test.csv', index=False)
   X_test = dataset_test.iloc[:, :4].values
   Y_test = dataset_test.iloc[:, -1].values
with Modelling:

   #tuning data
   n_steps = 5
   X, y = split_sequence(df_close, n_steps)
   n_steps = 5
   X, y = split_sequence(df_close, n_steps)  # column names to X and y data frames
   df_X = pd.DataFrame(X, columns=['t-' + str(i) for i in range(n_steps-1, -1, -1)])
   df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])

    # concat df_X and df_y
   df = pd.concat([df_X, df_y], axis=1)

   # Model knn
   # import knn
   from sklearn.neighbors import KNeighborsRegressor
   model_knn = KNeighborsRegressor(n_neighbors=7)
   model_knn.fit(X_train, y_train)
   y_pred=model_knn.predict(X_test)
   from sklearn.metrics import mean_squared_error
   mean_squared_error(y_test, y_pred)

   from sklearn.metrics import mean_absolute_percentage_error
   mean_absolute_percentage_error(y_test, y_pred)
   from sklearn.metrics import mean_absolute_error
   mean_absolute_error(y_test, y_pred)

   # Model naive bayes
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

    # Calculating MSE
   mse = mean_squared_error(y_test, y_pred)
   print("Mean Squared Error (MSE):", mse)

    # Calculating MAPE
   mape = mean_absolute_percentage_error(y_test, y_pred)
   print("Mean Absolute Percentage Error (MAPE):", mape)

   # Model Random Forest
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Create a Random Forest Regressor
   random_forest = RandomForestRegressor()

# Train the model
   random_forest.fit(X_train, y_train)

# Make predictions
   predictions = random_forest.predict(X_test)

# Calculate evaluation metrics
   mse = mean_squared_error(y_test, predictions)

   print("Mean Squared Error (MSE):", mse)

   mape = mean_absolute_percentage_error(y_test, predictions)
   print("Mean Absolute Percentage Error (MAPE):", mape)



with Implementasi:
   #menyimpan model
   with open('knn','wb') as r:
      pickle.dump(neigh,r)
   with open('minmax','wb') as r:
      pickle.dump(scaler,r)
   
   st.title("""Implementasi Data""")
   input_1 = st.number_input('Masukkan Data 1')
   input_2 = st.number_input('Masukkan Data 2')
   input_3 = st.number_input('Masukkan Data 3')
   input_4 = st.number_input('Masukkan Data 4')

   def submit():
      # inputs = np.array([inputan])
      with open('knn', 'rb') as r:
         model = pickle.load(r)
      with open('minmax', 'rb') as r:
         minmax = pickle.load(r)
      data1 = minmax.transform([[input_1]])
      data2 = minmax.transform([[input_2]])
      data3 = minmax.transform([[input_3]])
      data4 = minmax.transform([[input_4]])

      X_pred = model.predict([[(data1[0][0]),(data2[0][0]),(data3[0][0]),(data4[0][0])]])
      t_data1= X_pred.reshape(-1, 1)
      original = minmax.inverse_transform(t_data1)
      hasil =f"Prediksi Hasil Peramalan Pada Harga Penutupan Saham PLN adalah  : {original[0][0]}"
      st.success(hasil)

   all = st.button("Submit")
   if all :
      st.balloons()
      submit()

