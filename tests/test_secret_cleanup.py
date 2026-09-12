from pathlib import Path


def test_no_google_or_openai_secret_keys_or_branches_exist_in_app():
    app_text = Path("app.py").read_text()

    assert "GOOGLE_API_KEY" not in app_text
    assert "OPENAI_API_KEY" not in app_text
    assert "HUGGING_FACE_KEY" not in app_text
    assert "token=" not in app_text
    assert "provider == \"google\"" not in app_text
    assert "provider == \"openai\"" not in app_text
