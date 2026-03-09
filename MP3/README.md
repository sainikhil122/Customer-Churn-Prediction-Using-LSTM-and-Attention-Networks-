py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py