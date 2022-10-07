import boto3
from pprint import pprint

rds_client = boto3.client('rds')


response = rds_client.create_db_instance(
    DBName="rdstuts",
    DBInstanceIdentifier="rdstuts",
    AllocatedStorage=20,
    DBInstanceClass='db.t2.micro',
    Engine='MySQL',
    MasterUsername='admins',
    MasterUserPassword='password',
    Port=3306,
    EngineVersion='8.0.27',
    PubliclyAccessible=True,
    StorageType='gp2'
)

pprint(response)