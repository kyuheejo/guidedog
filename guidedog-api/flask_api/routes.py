from flask import Blueprint, request, make_response, jsonify
from ocr import detect_text
from text_to_speech import text_to_speech
from azure_image_captioning import image_captioning
from transfer import main

route_blueprint = Blueprint('route_blueprint', __name__)


@route_blueprint.route("/")
def index():
    return "Guide Dog API"


@route_blueprint.route("/ocr", methods=['POST'])
def ocr_text_to_speech():
    img = request.files["file"]
    texts = ' '.join(detect_text(img))
    audio = text_to_speech(texts)
    res = make_response(audio)
    res.headers['Content-Type'] = 'audio/mp3'
    return res


@route_blueprint.route("/ocr/text", methods=['POST'])
def ocr():
    img = request.files["file"]
    texts = ' '.join(detect_text(img))
    return jsonify(texts)


@route_blueprint.route("/caption", methods=['POST'])
def azure_image_captioning():
    img = request.files["file"]
    texts = image_captioning(img)
    return jsonify(texts)


@route_blueprint.route("/custom/caption", methods=['POST'])
def custom_image_captioning():
    img = request.files["file"].read()
    texts = main(img)
    return jsonify(texts)
