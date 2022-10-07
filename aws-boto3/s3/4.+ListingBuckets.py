import boto3



'''
bucket = boto3.client('s3')

response = bucket.list_buckets()

print("Listing all buckets")

for bucket in response['Buckets']:
    print(bucket['Name'])

'''

resource = boto3.resource('s3')

iterator = resource.buckets.all()

print("Listing all buckets")

for bucket in iterator:
    print(bucket.name)

