from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]/'app.py'


def test_estimate_and_example_profile_work():
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception
    app.button(key='FormSubmitter:estimate-Estimate charges').click().run()
    assert not app.exception
    before = app.metric[0].value
    assert before == '$5,100'
    app.button(key='load_example').click().run()
    assert not app.exception
    assert app.number_input(key='age').value == 44
    assert app.selectbox(key='smoker').value == 'yes'
    app.button(key='FormSubmitter:estimate-Estimate charges').click().run()
    assert not app.exception
    assert app.metric[0].value == '$25,101'
    assert app.metric[0].value != before
    assert any('21,184–29,018' in element.value for element in app.markdown)
    assert any('71.2%' in element.value for element in app.warning)


def test_last_result_is_labeled_and_persists_across_tabs():
    app = AppTest.from_file(APP, default_timeout=180).run()
    app.button(key='FormSubmitter:estimate-Estimate charges').click().run()
    before = app.metric[0].value
    app.run()
    assert not app.exception
    assert app.metric[0].value == before
    assert any('last submitted profile' in element.value for element in app.caption)
