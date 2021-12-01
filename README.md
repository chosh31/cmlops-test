# cmlops-test
- modify4

## Folder structure
- `.dvc`/
- `.github`/`workflows`/
- `models`/{model}/
	- `dvc` configs
		- `dvc.yaml`
		- `dvc.lock`
	- `docker`
		- `Dockerfile`
		- `docker-compose.yml`
	- `model` code
		- `requirements.txt`
		- `train.py`
		- `evaluate.py`
		- `infer.py`
	- `lambda` code
		- `lambda_handler.py`
