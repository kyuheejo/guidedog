from flask import Flask

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    from flask_api.routes import route_blueprint
    app.register_blueprint(route_blueprint)
    return app

