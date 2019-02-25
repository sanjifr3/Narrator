#!/usr/bin/env python3
"""File for launching narrator Flask app."""
from app import app

app.run(host="0.0.0.0", debug=True)
