from flask import Flask
from config import Config

# Make application named app
app = Flask(__name__)# , static_url_path='/static')
app.config.from_object(Config)

from app import views
