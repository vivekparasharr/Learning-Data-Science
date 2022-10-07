import boto3
from pprint import pprint

rds_client = boto3.client('rds')

response = rds_client.describe_db_instances(
    DBInstanceIdentifier = "rdstuts"
)

pprint(response)