FROM public.ecr.aws/lambda/python:3.9

RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY vegetable_classification_model.tflite .
COPY predict_lamda_func_dkr.py .

CMD ["predict_lamda_func_dkr.lambda_handler"]