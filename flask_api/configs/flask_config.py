from logging.handlers import TimedRotatingFileHandler


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    #SECRET_KEY = 'this-really-needs-to-be-changed'
    SERVER_PORT = 5000
    API_VERSION = '0.1.0'


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SERVER_ADDRESS: '0.0.0.0'
    SERVER_PORT: 5000


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    