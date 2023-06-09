import os
import azure.ai.vision as cvsdk
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load environment variables
load_dotenv()
endpoint = os.getenv('CV_ENDPOINT')
key = os.getenv('CV_KEY')

# Create a Vision Service
service_options = cvsdk.VisionServiceOptions(endpoint, key)

# Select an image to analyze
img_filename = "sample.jpg"
vision_source = cvsdk.VisionSource(filename=img_filename)

# Set image analysis options and features
analysis_options = cvsdk.ImageAnalysisOptions()
analysis_options.features = (
    cvsdk.ImageAnalysisFeature.CAPTION |
    cvsdk.ImageAnalysisFeature.DENSE_CAPTIONS |
    cvsdk.ImageAnalysisFeature.TAGS |
    cvsdk.ImageAnalysisFeature.OBJECTS
)

# Specify the language of the returned data
analysis_options.language = "en"

# Select gender neutral captions
analysis_options.gender_neutral_caption = True

# Get the Image Analysis results
image_analyzer = cvsdk.ImageAnalyzer(service_options, vision_source, analysis_options)
result = image_analyzer.analyze()

if result.reason == cvsdk.ImageAnalysisResultReason.ANALYZED:
    # Print caption
    if result.caption is not None:
        print(f"\nCaption: '{result.caption.content}' (Confidence {result.caption.confidence :.4f})")

    # Print dense captions
    if result.dense_captions is not None:
        print("\nDense Captions:\n")
        for caption in result.dense_captions:
            print(f" '{caption.content}' (Confidence: {caption.confidence :.4f})")

    # Print tags
    if result.tags is not None:
        print("\nTags:\n")
        for tag in result.tags:
            print(f" '{tag.name}' (Confidence {tag.confidence :.4f})")

    # Print objects
    if result.objects is not None:
        # Load a test image and get its dimensions
        img = Image.open(img_filename)
        img_height, img_width, img_ch = np.array(img).shape

        # Display the image
        draw = ImageDraw.Draw(img)

        # Select line width and color for the bounding box
        line_width = 3
        font_size = 18
        color = (0,255,0)

        print("\nObjects:\n")
        for object in result.objects:
            print(f" '{object.name}', {object.bounding_box} (Confidence: {object.confidence :.4f})")
            
            if object.confidence > 0.5:
                left = object.bounding_box.x
                top = object.bounding_box.y
                height = object.bounding_box.h
                width =  object.bounding_box.w
                # Create a rectangle
                shape = [(left, top), (left+width, top+height)]
                draw.rectangle(shape, outline=color, width=line_width)
                # Display probabilities
                font = ImageFont.truetype("arial.ttf", font_size)
                draw.text((left, top-20), f"{object.name} ({object.confidence * 100 :.2f}%)", fill=color, font=font)
        
        img.show()
        img.save("result.png", "PNG")
        print("Image saved!")

else:
    error_details = cvsdk.ImageAnalysisErrorDetails.from_result(result)
    print("Analysis failed.")
    print(f" Error reason: {error_details.reason}")
    print(f" Error code: {error_details.error_code}")
    print(f" Error message: {error_details.message}")