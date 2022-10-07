import boto3


client = boto3.client('s3')


with open('aws.png', 'rb') as f:
    data = f.read()


response = client.put_object(
    ACL="public-read-write",
    Bucket = "parwizforogh12",
    Body=data,
    Key='aws.png'

)

print(response)