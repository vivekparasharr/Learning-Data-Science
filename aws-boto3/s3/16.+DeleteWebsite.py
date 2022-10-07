import boto3

client = boto3.client('s3')

response = client.delete_bucket_website(
    Bucket="awstutorial12"
)

print(response)