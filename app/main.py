# app/main.py
from flask import Flask
from route.routes import first, second, config
from config.config import  load_config
    
CONFIG = load_config()

app = Flask(__name__)

def create_app():
    """
    应用工厂函数
    """
    app = Flask(__name__)
    app.register_blueprint(first, url_prefix='/1')
    app.register_blueprint(second, url_prefix='/2')
    app.register_blueprint(config, url_prefix='')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5009, debug=True)
    
    