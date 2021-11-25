import json
import logging
from infer import IrisONNXPredictor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

logger.info('Loading the model')
inferencing_instance = IrisONNXPredictor('./iris.onnx')

def lambda_handler(event, context):
	if 'resource' in event.keys():
		body = event['body']
		body = json.loads(body)
		params = body['params']
		logger.info(f'Got the input: {params}')

		response = inferencing_instance.predict(params)
		logger.info(json.dumps(response))
		return {
			'statusCode': 200,
			'headers': {},
			'body': json.dumps(response)
		}
	else:
		params = event["params"]
		logger.info(f'Got the input: {params}')
		response = inferencing_instance.predict(params)
		logger.info(response)
		return response

if __name__ == '__main__':
	test = {'params': [[5.1, 3.5, 1.4, 0.2],[5.1, 3.8, 1.9, 0.4],[4.9, 3.,  1.4, 0.2],[5.6, 2.8, 4.9, 2. ]]}
	lambda_handler(test, None)