from api.app import create_app
from configs.flask_config import ProductionConfig

application = create_app(config_object = ProductionConfig)


if __name__ == '__main__':
    application.run(host='0.0.0.0', use_reloader = False)
    