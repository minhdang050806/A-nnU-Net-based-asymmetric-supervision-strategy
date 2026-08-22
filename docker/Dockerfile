FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_compile=False \
    nnUNet_n_proc_DA=2 \
    nnUNet_def_n_proc=2 \
    nnUNet_raw=/opt/nnunet_raw \
    nnUNet_preprocessed=/opt/nnunet_preprocessed \
    nnUNet_results=/opt/nnunet_results

WORKDIR /opt/submission

COPY requirements.txt /opt/submission/requirements.txt
RUN pip install --no-cache-dir -r /opt/submission/requirements.txt

COPY nnUNet /opt/nnUNet
RUN pip install --no-cache-dir --no-deps /opt/nnUNet

COPY models /opt/nnunet_results
COPY run_model.py /opt/submission/run_model.py

ENTRYPOINT ["python", "/opt/submission/run_model.py"]
