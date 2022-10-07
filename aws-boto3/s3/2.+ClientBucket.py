import boto3


client = boto3.client('s3')

response = client.create_bucket(
    Bucket = "parwizforogh12",
    ACL = "private",

)

print(response)