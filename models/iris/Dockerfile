FROM amazon/aws-lambda-python

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

COPY . /app

WORKDIR /app
# RUN cd ./models/iris

# RUN pip install -r requirements.txt --no-cache-dir
RUN pip uninstall protobuf python3-protobuf
RUN pip install --upgrade pip
RUN pip install --upgrade protobuf
RUN pip install -r requirements.txt
# RUN pip install "dvc[s3]"

RUN dvc init --no-scm
RUN dvc remote add -d storage s3://knowre-datacentral/test/dvc

# RUN dvc pull models/iris/iris.dvc
RUN dvc repro
RUN python lambda_handler.py

RUN chmod -R 0755 lambda_handler.py

CMD [ "lambda_handler.lambda_handler" ]
