from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
from medcost.data import FEATURES

st.set_page_config(page_title='MedCost Explorer', page_icon='📊', layout='wide')
st.title('MedCost Explorer')
st.write('Explore an educational model of historical medical charges.')
st.warning('This is a portfolio demonstration, not an insurance quote, premium calculator, clinical tool, or validated estimate of present-day costs. Source collection dates and population representativeness are unverified.')

@st.cache_resource
def load_model():
    path = Path(__file__).parent/'artifacts/model.pkl'
    if not path.exists():
        from medcost.training import main
        main()
    # Only load the artifact produced by this repository in a trusted environment.
    with path.open('rb') as handle:
        return pickle.load(handle)

bundle = load_model()
with st.form('estimate'):
    a,b = st.columns(2)
    age = a.number_input('Age (dataset range: 18–64)', 18, 64, 35)
    bmi = a.number_input('BMI (dataset range: approximately 16–53)', 15.96, 53.13, 28.0)
    children = a.number_input('Covered children/dependents', 0, 5, 0)
    sex = b.selectbox('Sex (source dataset categories)', ['female','male'])
    smoker = b.selectbox('Smoking status (self-reported input)', ['no','yes'])
    region = b.selectbox('US region', ['northeast','northwest','southeast','southwest'])
    submitted = st.form_submit_button('Estimate historical charges')
if submitted:
    row = pd.DataFrame([dict(age=age,bmi=bmi,children=children,sex=sex,smoker=smoker,region=region)])[FEATURES]
    prediction = float(bundle['model'].predict(row)[0])
    radius = bundle['radius']
    st.metric('Estimated historical charges', f'${prediction:,.0f}')
    st.write(f'Nominal 90% calibrated range: **${max(0,prediction-radius):,.0f}–${prediction+radius:,.0f}**')
    st.caption('This interval is calibrated on a held-out sample. Its interpretation assumes comparable, exchangeable data; coverage for an individual, subgroup, or present-day population is not guaranteed. No input is sent to a database by this app.')
with st.expander('How was this evaluated?'):
    st.json(bundle['metrics'])
    st.write('Five-fold cross-validation on the fitting partition selects the model. Separate observations calibrate intervals; final test observations measure performance. Preprocessing is learned inside each training fold. The model does not infer smoking status.')
st.caption('Input bounds reflect dataset coverage, not clinical eligibility. Associations and model explanations do not establish causal effects.')
