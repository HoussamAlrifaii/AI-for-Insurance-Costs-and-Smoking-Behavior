from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_estimate_renders_and_input_change_updates_prediction():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=180).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    before=app.metric[0].value
    app.selectbox[1].set_value('yes')
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value!=before
