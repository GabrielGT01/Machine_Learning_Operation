

import predict

ride = {
    "passenger_count":1.0,
    "trip_distance": 4.12,
    "fare_amount":21.20,
    "total_amount":36.77,
    "PULocationID": 171,
    "DOLocationID": 73,
    
}

time = predict.predict_from_dict(ride)

if time is not None:
    print(f"Predicted duration: {time:.2f} minutes")
else:
    print("Prediction failed.")

