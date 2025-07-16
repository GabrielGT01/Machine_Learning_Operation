

import predict

ride = {
    "PULocationID": 75,
    "DOLocationID": 40,
    "trip_distance": 5
}

time = predict.predict_from_dict(ride)

if time is not None:
    print(f"Predicted duration: {time:.2f} minutes")
else:
    print("Prediction failed.")

