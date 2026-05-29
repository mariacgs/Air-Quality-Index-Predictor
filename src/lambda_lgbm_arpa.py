
import json
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("arpa_aqi_model.onnx")

def get_eaqi_label(level):
    """Maps the integer level to the official EEA health category."""
    labels = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
        6: "Extremely Poor"
    }
    return labels.get(max(1, min(6, round(level))), "Unknown")

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        
        # Expected feature order (**must match input schema**):
            #f0: CO_mean
            #f1: NO2_max
            #f2: O3_max
            #f3: O3_mean
            #f4: PM10_mean
            #f5: PM2.5_mean
            #f6: SO2_max
            #f7: SO2_mean
            #f8: year
            #f9: month
            #f10: day
            #f11: day_of_week
        features = np.array(body['features'], dtype=np.float32).reshape(1, -1)
        
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: features})[0]
        
        raw_val = float(prediction[0])
        final_level = int(max(1, min(6, round(raw_val))))
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  
            },
            'body': json.dumps({
                'model': 'XGBoost-EAQI-v1',
                'continuous_score': round(raw_val, 4),
                'eaqi_level': final_level,
                'status': get_eaqi_label(final_level),
                'city': 'Rome',
                'standard': 'EEA (European Environment Agency)'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f"Inference Failed: {str(e)}"})
        }
