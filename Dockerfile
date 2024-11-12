FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /workspace
RUN rm -rf *
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/* ./

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=*", "--port=8888", "--no-browser", "--allow-root","--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'", "--notebook-dir=/"]