from flask import Flask
from app import app


def test_base_route():
    client = app.test_client()
    url = '/'

    response = client.get(url)
    assert response.status_code == 200

def test_post():
    client = app.test_client()
    url = '/predict'

    response = client.get(url)
    assert response.status_code == 405



