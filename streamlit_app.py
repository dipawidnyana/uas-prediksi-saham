import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
import altair as alt
import plotly.graph_objs as go

# Fungsi untuk mengunduh dan mempersiapkan data
@st.cache
def load_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    closing_prices = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_closing_prices = scaler.fit_transform(closing_prices)
    return stock_data, closing_prices, scaled_closing_prices, scaler

# Fungsi untuk menyiapkan data untuk model RNN
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Fungsi untuk membangun model
def build_model(model_type, time_steps):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=64, activation='relu', input_shape=(time_steps, 1)))
    elif model_type == 'GRU':
        model.add(GRU(units=64, activation='relu', input_shape=(time_steps, 1)))
    else:
        model.add(SimpleRNN(units=64, activation='relu', input_shape=(time_steps, 1)))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Fungsi untuk memprediksi harga saham
def predict_future_prices(model, data, time_steps, future_days):
    predictions = []
    last_sequence = data[-time_steps:].reshape((1, time_steps, 1))

    for _ in range(future_days):
        next_price = model.predict(last_sequence)
        predictions.append(next_price[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], np.array(next_price).reshape(1, 1, 1), axis=1)

    return np.array(predictions)

# Main menu
st.title('Prediksi Harga Saham')

# Input di sidebar
st.sidebar.header('Pilih Saham')
ticker = st.sidebar.text_input('Masukkan ticker saham (contoh: PTBA.JK):', 'PTBA.JK')
start_date = st.sidebar.date_input('Tanggal mulai:', pd.to_datetime('2021-03-23'))
end_date = st.sidebar.date_input('Tanggal akhir:', pd.to_datetime('2024-04-23'))
future_days = st.sidebar.number_input('Masukkan jumlah hari prediksi:', min_value=1, max_value=365, value=60)
model_type = st.sidebar.selectbox('Pilih jenis model:', ['LSTM', 'GRU', 'RNN'])
run_button = st.sidebar.button('Run')

if run_button and ticker and start_date and end_date and future_days:
    # Informasi saham
    info = yf.Ticker(ticker).info
    st.subheader('Informasi Saham')
    st.write(f"Nama: {info.get('longName', 'N/A')}")
    st.write(f"Sektor: {info.get('sector', 'N/A')}")
    st.write(f"Industri: {info.get('industry', 'N/A')}")
    st.write(f"Website: {info.get('website', 'N/A')}")

    # Mengunduh data historis harga saham
    stock_data, closing_prices, scaled_closing_prices, scaler = load_data(ticker, start_date, end_date)

    # Tampilkan candlestick chart
    st.subheader(f'Candlestick chart untuk {ticker}')
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])
    fig.update_layout(title=f'Candlestick chart for {ticker}', yaxis_title='Stock Price')
    st.plotly_chart(fig)

    # Menghilangkan langkah-langkah di main menu setelah tombol Run ditekan
    st.markdown("### Memuat prediksi...")

    # Membagi data menjadi data pelatihan dan pengujian
    train_size = int(len(scaled_closing_prices) * 0.8)
    train_data = scaled_closing_prices[:train_size]
    test_data = scaled_closing_prices[train_size:]

    # Menyiapkan data untuk model RNN
    time_steps = 30
    X_train, y_train = prepare_data(train_data, time_steps)
    X_test, y_test = prepare_data(test_data, time_steps)

    # Membangun dan melatih model
    model = build_model(model_type, time_steps)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Prediksi harga saham masa depan untuk future_days hari ke depan
    future_predictions = predict_future_prices(model, scaled_closing_prices, time_steps, future_days)
    future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

    # Hapus pesan "Memuat prediksi..."
    st.markdown("### Prediksi selesai!")

    # Tampilkan grafik prediksi dengan Altair
    st.subheader(f'Prediksi harga saham untuk {future_days} hari ke depan')
    future_dates = pd.date_range(start=stock_data.index[-1], periods=future_days + 1, closed='right')
    predicted_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

    # Tampilkan data historis dan prediksi dengan Altair
    historical_data = stock_data.reset_index()
    predicted_data = predicted_data.reset_index()

    historical_chart = alt.Chart(historical_data).mark_line().encode(
        x='Date',
        y='Close',
        tooltip=['Date', 'Close']
    ).properties(
        width=800,
        height=400
    ).interactive()

    predicted_chart = alt.Chart(predicted_data).mark_line(color='orange').encode(
        x='index:T',
        y='Predicted Close:Q',
        tooltip=['index:T', alt.Tooltip('Predicted Close:Q', format='.2f')],
    ).properties(
        width=800,
        height=400
    ).interactive()

    st.altair_chart(historical_chart + predicted_chart, use_container_width=True)

    # Tampilkan angka perkiraan per 10 hari
    st.subheader('Harga prediksi per 10 hari')
    for i in range(0, future_days, 10):
        if i + 10 < future_days:
            st.write(f"Harga untuk hari ke-{i + 10}: {future_predictions[i + 9][0]:.2f}")
        else:
            st.write(f"Harga untuk hari ke-{i + future_days % 10}: {future_predictions[-1][0]:.2f}")

    # Tampilkan angka terakhir dari prediksi saham
    st.write(f"Prediksi harga saham terakhir untuk {ticker} adalah {future_predictions[-1][0]:.2f}")

    # Download hasil prediksi
    csv = predicted_data.to_csv(index=False)
    st.download_button('Unduh hasil prediksi', csv, 'predicted_data.csv', 'text/csv')
else:
    st.markdown("""
    Langkah-langkah penggunaan:
    1. Klik tombol sidebar yang ada di pojok kiri atas 
    2. Masukkan ticker saham (contoh: PTBA.JK).
    3. Tentukan rentang waktu dengan memilih tanggal mulai dan akhir.
    4. Masukkan jumlah hari untuk prediksi harga saham ke depan.
    5. Tekan tombol **Run** untuk memulai prediksi.
    """)
