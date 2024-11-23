FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /workspace
RUN rm -rf *
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install jupyter_kernel_gateway
RUN pip install -r requirements.txt
RUN pip install

COPY src/* ./

EXPOSE 8888
# CMD ["jupyter", "lab", "--ip=*", "--port=8888", "--no-browser", "--allow-root","--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'", "--notebook-dir=/"]
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]