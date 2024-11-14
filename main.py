import boto3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

def detect_labels(photo, bucket):
    #initialize the Rekognition and S3 clients
    rekognition_client = boto3.client('rekognition')
    s3_client = boto3.resource('s3')

    try:
        #detect labels using Amazon Rekognition
        response = rekognition_client.detect_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
            MaxLabels=10
        )
    except Exception as e:
        print("Error detecting labels:", e)
        return 0

    #display detected labels
    print(f'Detected labels for {photo}:\n')
    for label in response['Labels']:
        print(f"Label: {label['Name']}")
        print(f"Confidence: {label['Confidence']}\n")

    #load and display the image with bounding boxes
    try:
        img_data = s3_client.Object(bucket, photo).get()['Body'].read()
        img = Image.open(BytesIO(img_data))
    except Exception as e:
        print("Error loading image from S3:", e)
        return 0

    #set up plot
    plt.imshow(img)
    ax = plt.gca()

    #draw bounding boxes for detected instances
    for label in response['Labels']:
        for instance in label.get('Instances', []):
            bbox = instance['BoundingBox']
            left = bbox['Left'] * img.width
            top = bbox['Top'] * img.height
            width = bbox['Width'] * img.width
            height = bbox['Height'] * img.height
            rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label_text = f"{label['Name']} ({label['Confidence']:.2f}%)"
            plt.text(left, top - 2, label_text, color='r', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.show()
    return len(response['Labels'])

def main():
    photo = 'pedestrians-crossing-road-street-preview.jpg'
    bucket = 'aws-rekognition-label-images-thanhsang-v11'
    label_count = detect_labels(photo, bucket)
    print("Labels detected:", label_count)

if __name__ == "__main__":
    main()
