FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /workspace
RUN rm -rf *
RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir jupyter_kernel_gateway
COPY requirements.txt cuda-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r cuda-requirements.txt

COPY src/* ./

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]