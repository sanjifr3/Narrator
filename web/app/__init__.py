"""
Define flask app and import config.
"""
from flask import Flask
from config import Config

# Make application named app
app = Flask(__name__)  # , static_url_path='/static')
app.config.from_object(Config)

# Assign secret key
app.secret_key = app.config['SECRET_KEY']

from app import views
