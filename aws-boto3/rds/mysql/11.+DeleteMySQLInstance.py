import boto3

rds_client = boto3.client('rds')

response = rds_client.delete_db_instance(
    DBInstanceIdentifier="rdstuts",
    SkipFinalSnapshot=False,
    FinalDBSnapshotIdentifier="rdstuts-final-snapshot",
    DeleteAutomatedBackups=True

)

print(response)