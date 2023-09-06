import os
from flask import Flask
import config

app = Flask(__name__)

if os.environ.get('FLASK_ENV') == 'development':
    app.config.from_object(config.DevelopmentConfig)
elif os.environ.get('FLASK_ENV') == 'production':
    app.config.from_object(config.ProductionConfig)
