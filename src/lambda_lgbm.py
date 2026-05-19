import json
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("lgbm_aqi_model.onnx")

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        
        input_data = np.array(body['features'], dtype=np.float32).reshape(1, -1)
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: input_data})[0]
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'model': 'LightGBM',
                'predicted_aqi': round(float(prediction[0]), 2)
            })
        }
    except Exception as e:
        return {'statusCode': 400, 'body': json.dumps({'error': str(e)})}