FROM levp21/pytorch-base:latest
WORKDIR /web_app

ENV PIP_REQUIRE_HASHES=0

COPY app/main.py /web_app/
COPY requirements.txt /web_app/
COPY weights/person_det_model.pt /web_app/
COPY weights/ppe_best.pt /web_app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "main.py"]

EXPOSE 8501