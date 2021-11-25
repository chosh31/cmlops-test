import onnxruntime as ort

class IrisONNXPredictor:
	def __init__(self, model_path):
		self.ort_session = ort.InferenceSession(model_path)

	def predict(self, args):
		input_name = self.ort_session.get_inputs()[0].name
		return self.ort_session.run(None, {input_name: args})[0]

if __name__ == "__main__":
	params = [[5.1, 3.5, 1.4, 0.2],[5.1, 3.8, 1.9, 0.4],[4.9, 3.,  1.4, 0.2],[5.6, 2.8, 4.9, 2. ]]
	predictor = IrisONNXPredictor("./iris.onnx")
	print(predictor.predict(params))
