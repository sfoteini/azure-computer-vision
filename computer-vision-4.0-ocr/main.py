import os
import azure.ai.vision as cvsdk
from dotenv import load_dotenv
from PIL import Image, ImageDraw
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

# To analyze an online image, uncomment the following lines
#url = "<YOUR_URL>"
#vision_source = cvsdk.VisionSource(url=url)

# Set image analysis options and features
analysis_options = cvsdk.ImageAnalysisOptions()
analysis_options.features = (
    cvsdk.ImageAnalysisFeature.TEXT
)

# Get the Image Analysis results
image_analyzer = cvsdk.ImageAnalyzer(service_options, vision_source, analysis_options)
result = image_analyzer.analyze()

if result.reason == cvsdk.ImageAnalysisResultReason.ANALYZED:
    # Print extracted text
    if result.text is not None:
        # Load a test image and get its dimensions
        img = Image.open(img_filename)
        img_height, img_width, img_ch = np.array(img).shape

        # Display the image
        draw = ImageDraw.Draw(img)

        # Select line width and color for the bounding box
        line_width = 5
        color = (255,255,255)

        print("\nText:\n")

        for line in result.text.lines:
            print(f"  Line: '{line.content}'")

            # Create a rectangle
            draw.polygon(line.bounding_polygon, outline=color, width=line_width)

            for word in line.words:
                print(f"    Word: '{word.content}': Confidence {word.confidence :.4f}")

        img.show()
        img.save("result.png", "PNG")
        print("Image saved!")

else:
    error_details = cvsdk.ImageAnalysisErrorDetails.from_result(result)
    print("Analysis failed.")
    print(f" Error reason: {error_details.reason}")
    print(f" Error code: {error_details.error_code}")
    print(f" Error message: {error_details.message}")