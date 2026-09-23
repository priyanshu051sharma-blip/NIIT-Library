FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
RUN rm -rf /usr/local/lib/python3.12/site-packages/cv2 /usr/local/lib/python3.12/site-packages/opencv_python* /usr/local/lib/python3.12/site-packages/opencv_contrib_python*
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84
COPY app ./app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
