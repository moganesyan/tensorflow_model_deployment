from api.app import create_app
from configs.flask_config import ProductionConfig

app = create_app(config_object = ProductionConfig)


if __name__ == '__main__':
    app.run()
    