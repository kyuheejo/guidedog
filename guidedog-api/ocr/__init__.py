def detect_text(img):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img.read())

    response = client.text_detection(image=image)
    texts = response.text_annotations
    texts = [text.description for text in texts[1:]]
    print(" ".join(texts))
    return texts
