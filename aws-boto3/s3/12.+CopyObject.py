
import boto3

s3 = boto3.resource('s3')

copy_source = {
    'Bucket':'parwizforogh7777',
    'Key':'file.pdf'
}

s3.meta.client.copy(copy_source, 'parwizforogh12', 'copied.pdf')