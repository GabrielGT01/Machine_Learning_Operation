import predict

ride = {
    "passenger_count": 1.0,
    "trip_distance": 5.93,
    "fare_amount": 24.70,
    "total_amount": 34.00,
    "PULocationID": 75,
    "DOLocationID": 235,
}

time = predict.predict_from_dict(ride)

if time is not None:
    print(f"Predicted duration: {time:.2f} minutes")
else:
    print("Prediction failed.")
