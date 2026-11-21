from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200


def test_ask_without_upload_gives_error():
    response = client.get("/ask", params={"question": "hello"})
    assert response.status_code == 400


def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400