from flask import Flask
from configs.logging_config import get_logger


_logger = get_logger(logger_name = __name__)


def create_app(config_object) -> Flask:
	"""Create a flask app instance."""
	#_logger.debug('create_app Called')

	flask_app = Flask('coco_detector')
	flask_app.config.from_object(config_object)

	from api.controller import coco_detector
	flask_app.register_blueprint(coco_detector)
	_logger.debug('Application instance created')

	return flask_app