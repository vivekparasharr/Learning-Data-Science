import boto3
from pprint import pprint


client = boto3.client('s3')

response = client.get_bucket_website(
    Bucket="awstutorial12"
)

pprint(response)