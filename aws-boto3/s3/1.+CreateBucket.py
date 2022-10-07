import boto3



bucket = boto3.resource('s3')

response = bucket.create_bucket(
    Bucket = "parwizforogh7777",
    ACL="private",



)

print(response)