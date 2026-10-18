from pathlib import Path
import pandas as pd
import streamlit as st
from medcost.data import FEATURES

st.set_page_config(page_title='MedCost Explorer | Medical charge estimates', page_icon='📊', layout='wide')
DEFAULT = dict(age=35, bmi=28.0, children=0, sex='female', smoker='no', region='northeast')
EXAMPLE = dict(age=44, bmi=28.0, children=0, sex='female', smoker='yes', region='northeast')
for name, value in DEFAULT.items():
    if name not in st.session_state:
        st.session_state[name] = value


def load_example():
    st.session_state.update(EXAMPLE)
    st.session_state.pop('result', None)
    st.session_state['example_loaded'] = True


@st.cache_resource
def load_model():
    from medcost.calibrate_log import build_bundle
    return build_bundle()


@st.cache_data
def evaluation():
    return pd.read_csv(Path(__file__).parent/'reports/log_test_comparison.csv')


st.caption('A DATA SCIENCE PORTFOLIO PROJECT')
st.title('Understand medical charges. Explore an estimate.')
st.write('MedCost Explorer uses a few profile details to estimate medical charges learned from a historical dataset. See the estimate, explore its uncertainty, and learn how the model works.')
st.info('Educational demonstration. Estimates are not insurance premiums, medical quotes, or validated present-day costs.')

estimate_tab, explanation_tab, performance_tab = st.tabs(['Estimate charges', 'How it works', 'Results & limitations'])

with estimate_tab:
    left, right = st.columns([1.6, 1], gap='large')
    with left:
        st.subheader('1. Build a profile')
        st.write('Enter a profile, then select **Estimate charges**. Use an illustrative example if you want to explore first.')
        st.button('Try an example profile', on_click=load_example, key='load_example')
        if st.session_state.get('example_loaded'):
            st.caption('Illustrative example: age 44, BMI 28, female, smoker, no dependents, northeast. It has no known actual bill and was not added to training data. You can edit any field.')
        with st.form('estimate'):
            a, b = st.columns(2)
            age = a.number_input('Age', 18, 64, key='age', help='The dataset covers ages 18–64. These bounds describe data coverage, not eligibility.')
            bmi = a.number_input('Body mass index (BMI)', 15.96, 53.13, step=.1, key='bmi', help='BMI is weight in kilograms divided by height in metres squared. Dataset range: approximately 16–53.')
            children = a.number_input('Covered children / dependents', 0, 5, key='children')
            sex = b.selectbox('Sex', ['female', 'male'], key='sex', format_func=str.title, help='These are the categories available in the source dataset.')
            smoker = b.selectbox('Smoking status', ['no', 'yes'], key='smoker', format_func=lambda value: 'Non-smoker' if value == 'no' else 'Smoker', help='An input you provide. The app does not predict whether someone smokes.')
            region = b.selectbox('US region', ['northeast','northwest','southeast','southwest'], key='region', format_func=str.title)
            submitted = st.form_submit_button('Estimate charges', type='primary', use_container_width=True)
        st.caption('Use example profiles rather than identifying medical information. The app does not write your inputs to a database.')
        if submitted:
            profile = dict(age=age, bmi=bmi, children=children, sex=sex, smoker=smoker, region=region)
            with st.spinner('Preparing your estimate…'):
                bundle = load_model()
                prediction = float(bundle['model'].predict(pd.DataFrame([profile])[FEATURES])[0])
            st.session_state['result'] = dict(profile=profile, prediction=prediction, radius=bundle['radius'])
    with right:
        st.subheader('2. Read your estimate')
        result = st.session_state.get('result')
        with st.container(border=True):
            if result is None:
                st.write('Your estimate will appear here.')
                st.caption('Complete the profile or try the example, then select Estimate charges.')
            else:
                prediction, radius = result['prediction'], result['radius']
                st.metric('Estimated historical charges', f'${prediction:,.0f}')
                st.markdown(f'**Nominal 90% prediction range**\n\nUSD **{max(0, prediction-radius):,.0f}–{prediction+radius:,.0f}**')
                st.write('The amount is a model estimate, not an actual bill. The range describes uncertainty; recorded charges can fall outside it.')
                st.caption('90% is an overall coverage target for comparable observations, not a guarantee for an individual or subgroup.')
                st.write('**Profile used for this estimate**')
                p = result['profile']
                st.caption(f"Age {p['age']} · BMI {p['bmi']:.1f} · {p['sex'].title()} · {'Smoker' if p['smoker']=='yes' else 'Non-smoker'} · {p['children']} dependents · {p['region'].title()}")
                st.caption('Results refer to the last submitted profile. After editing inputs, select Estimate charges again.')
        st.warning('High charges are harder to estimate. Among test records with actual charges of USD 20,000 or more, prediction ranges covered 71.2%, below the 90% overall target. The actual charge group is unknown when predicting.')

with explanation_tab:
    st.subheader('From profile to prediction')
    for column, heading, body in zip(st.columns(3),
            ['1 · Describe a profile', '2 · Get an estimate', '3 · Interpret the range'],
            ['Provide six inputs: age, BMI, dependents, sex, smoking status and US region.',
             'A random forest learns patterns from historical records. It estimates log-transformed charges, then converts the result back into dollars.',
             'Separate calibration records determine a margin around each estimate. A range communicates uncertainty rather than an exact future bill.']):
        with column:
            with st.container(border=True):
                st.markdown(f'**{heading}**')
                st.write(body)
    st.subheader('An example, explained')
    st.write('For the illustrative profile above, our tested environment produced an estimate of about USD 25,101 and a range of USD 21,184–29,018. This is an example prediction, not an observed medical bill or an accuracy check.')
    st.write('Changing one input while holding the others fixed explores how the model responds. It does not establish that changing that factor would cause the same change in someone’s real costs.')
    with st.expander('What was the model trained on?'):
        st.write('The CSV contains 1,338 rows and seven columns. One exact duplicate is removed. Missing BMI values are filled using training-only statistics. We use 801 fitting records, 268 calibration records and 268 test records.')
        st.write('The original project links the data to Kaggle. Original collection dates, billing period, authenticity and population representativeness have not been independently verified. We do not label these as annual costs or adjust them to current prices.')
    st.caption('Model: log-target random forest. The first estimate fits and calibrates the fixed model; later requests reuse it. No model selection or test evaluation is performed from your inputs.')

with performance_tab:
    st.subheader('What the evaluation tells us')
    st.write('The log-target model improved overall absolute error across repeated cross-validation splits. It still makes large mistakes, especially on expensive records.')
    selected = evaluation().loc[lambda frame: frame.model == 'Log target'].iloc[0]
    a, b, c = st.columns(3)
    a.metric('Average absolute error', f'${selected.MAE_USD:,.0f}')
    b.metric('Estimates within USD 2,000', f'{selected.within_2000_pct:.1f}%')
    c.metric('Test range coverage', f'{selected.coverage_pct:.1f}%')
    st.caption(f'Follow-up comparison on {int(selected.test_rows)} previously inspected test records. These are exploratory results, not fresh independent confirmation or accuracy guarantees.')
    st.markdown('**Overall results do not describe every group equally.**')
    st.dataframe(pd.DataFrame({
        'Recorded charge group': ['Below USD 10,000', 'USD 10,000–19,999', 'USD 20,000 or more'],
        'Mean absolute error (USD)': [877, 1819, 4144],
        'Prediction-range coverage': ['98.6%', '91.3%', '71.2%']
    }), hide_index=True, use_container_width=True)
    st.write('These groups use the known recorded charges for retrospective analysis. The app cannot identify the actual group for a new profile in advance.')
    with st.expander('Evaluation method and limitations'):
        st.write('Candidate models were compared using five-fold cross-validation within fitting records, with preprocessing fitted separately in each fold. The log-target extension was then compared across five fold arrangements. A separate calibration sample produced the approximately USD 3,917 range margin.')
        st.write('The research direction followed earlier test diagnostics, so the follow-up test results are exploratory. They do not establish performance on a genuinely new dataset. No fabricated records were added.')
        st.write('The dataset lacks treatment details and medical history. Its coverage is limited, and predictions are not clinical recommendations, insurance premiums or validated current-day estimates. A range targets marginal coverage under exchangeability; it does not guarantee individual or subgroup coverage.')
    st.link_button('Explore the code and research', 'https://github.com/HoussamAlrifaii/AI-for-Insurance-Costs-and-Smoking-Behavior')

st.divider()
st.caption('MedCost Explorer · Educational machine learning project · Dollar values use source units; collection dates and billing period are unverified.')
