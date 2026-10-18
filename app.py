from pathlib import Path
import pandas as pd
import streamlit as st
from medcost.data import FEATURES

st.set_page_config(page_title='MedCost Explorer', page_icon='📊', layout='wide')
st.title('MedCost Explorer')
st.write('Explore an educational model of historical medical charges.')
st.warning('This is a portfolio demonstration, not an insurance quote, premium calculator, clinical tool, or validated estimate of present-day costs. Source collection dates and population representativeness are unverified.')

@st.cache_resource
def load_model():
    # Fit only the fixed selected candidate; do not rerun model selection or test scoring.
    from medcost.calibrate_log import build_bundle
    return build_bundle()

bundle = load_model()
st.caption('Model: log-target random forest. Estimates are converted back to dollars. The first run fits and calibrates the fixed model; later requests reuse it.')
st.info('Expensive cases remain difficult. In the exploratory test comparison, average error was about USD 4,144 for recorded charges of USD 20,000 or more, and prediction ranges covered only 71.2% of that group. The app cannot know someone’s actual charge group in advance.')
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
    st.write(f'Nominal 90% calibrated range: **USD {max(0,prediction-radius):,.0f}–{prediction+radius:,.0f}**')
    st.caption('This interval is calibrated on a held-out sample. Its interpretation assumes comparable, exchangeable data; coverage for an individual, subgroup, or present-day population is not guaranteed. No input is sent to a database by this app.')
with st.expander('How was this evaluated?'):
    comparison = pd.read_csv(Path(__file__).parent/'reports/log_test_comparison.csv')
    selected = comparison.loc[comparison.model == 'Log target'].iloc[0]
    st.write(f"Follow-up test comparison: MAE USD {selected.MAE_USD:,.0f}; {selected.within_2000_pct:.1f}% of estimates within USD 2,000; range coverage {selected.coverage_pct:.1f}% across {int(selected.test_rows)} records.")
    st.write('These are exploratory results on an already inspected test set, not fresh independent confirmation. The original model was selected with five-fold cross-validation; the log-target extension was compared across repeated fitting-data folds. Separate records calibrate its prediction range. Preprocessing is learned within training folds. Smoking status is an input, not a predicted diagnosis.')
    st.write('801 fitting records · 268 calibration records · 268 test records. Calibration margin is approximately USD 3,917. Nominal 90% coverage is an overall target, not a guarantee for each person or subgroup.')
st.caption('Input bounds reflect dataset coverage, not clinical eligibility. Associations and model explanations do not establish causal effects.')
