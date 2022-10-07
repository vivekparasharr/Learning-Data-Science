
import boto3
import json

# topics covered
# create users
# list users
# update user
# create policy - careful as this example is an admin access policy
# list all policies
# attach policy to a user
# detach policy
# create user groups
# attach policy to user groups
# add users
# detach policy from group
# create access key
# deactivate access key
# create login profile for programatic access
# delete user
# delete user from user group

# create users
def create_user(username):
    iam=boto3.client('iam')
    response = iam.create_user(UserName=username)
    print(response)
create_user('testuser_rds')

# list users
def all_users():
    iam=boto3.client('iam')
    paginator = iam.get_paginator('list_users')
    for response in paginator.paginate():
        for user in response['Users']:
            username = user['UserName']
            Arn = user['Arn']
            print('Username: {} Arn: {}'.format(username, Arn))
all_users()

# update user
def update_user(old_username, new_username):
    iam = boto3.client('iam')
    response = iam.update_user(
        UserName=old_username,
        NewUserName=new_username
    )
    print(response)
update_user('testuser1', 'testuser')

# create policy - careful as this example is an admin access policy
def create_policy():
    iam = boto3.client('iam')
    user_policy = {
        "Version":"2012-10-17",
        "Statement":[
            {
                "Effect": "Allow",
                "Action": "*",
                "Resource": "*"
            }
        ]
    }
    response = iam.create_policy(
        PolicyName = 'pyFullAccess',
        PolicyDocument=json.dumps(user_policy)
    )
    print(response)
create_policy()

# list all policies
def list_policies():
    iam = boto3.client('iam')
    paginator = iam.get_paginator('list_policies')
    for response in paginator.paginate(Scope="AWS"):
        for policy in response['Policies']:
            policy_name = policy['PolicyName']
            Arn = policy['Arn']
            print('Policy Name : {} Arn : {}'.format(policy_name, Arn))
list_policies()

# attach policy to a user
def attach_policy(policy_arn, username):
    iam = boto3.client('iam')
    response = iam.attach_user_policy(
        UserName = username,
        PolicyArn = policy_arn
    )
    print(response)
attach_policy('arn:aws:iam::001815726351:policy/pyFullAccess', 'testuser')
attach_policy('arn:aws:iam::aws:policy/AmazonRDSFullAccess', 'testuser_rds')

# detach policy
iam = boto3.client('iam')
response = iam.detach_user_policy(
    UserName = 'testuser',
    PolicyArn = 'arn:aws:iam::001815726351:policy/pyFullAccess'
)
print(response)

# create user groups
def create_group(group_name):
    iam = boto3.client('iam')
    iam.create_group(GroupName=group_name)
create_group('S3Admins')

# attach policy to user groups
def attach_policy(policy_arn, group_name):
    iam = boto3.client('iam')
    response = iam.attach_group_policy(
        GroupName=group_name,
        PolicyArn=policy_arn
    )
    print(response)
attach_policy('arn:aws:iam::aws:policy/AmazonS3FullAccess','S3Admins')

# add users
def add_user(username, group_name):
    iam = boto3.client('iam')
    response = iam.add_user_to_group(
        UserName=username,
        GroupName=group_name
    )
    print(response)
add_user('testuser_rds', 'S3Admins')

# detach policy from group
iam = boto3.client('iam')
iam.detach_group_policy(
    GroupName='S3Admins',
    PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
)
print(response)

# create access key
def create_access(username):
    iam = boto3.client('iam')
    response = iam.create_access_key(
        UserName=username
    )
    print(response)
create_access('testuser')

# deactivate access key
def update_access():
    iam = boto3.client('iam')
    iam.update_access_key(
        AccessKeyId='paste-access-key-here',
        Status='Inactive',
        UserName='testuser'
    )
update_access()

# create login profile for programatic access
def create_login(username):
    iam = boto3.client('iam')
    login_profile = iam.create_login_profile(
        Password = 'Mypassword@1',
        PasswordResetRequired = False,
        UserName = username
    )
    print(login_profile)
create_login('test')

# delete user
def delete_myuser(username):
    iam = boto3.client('iam')
    response = iam.delete_user(
        UserName=username
    )
    print(response)
delete_myuser('test')

# delete user from user group
def delete_user_group(username, groupName):
    iam = boto3.resource('iam')
    group = iam.Group(groupName)
    response = group.remove_user(
        UserName=username
    )
    print(response)
delete_user_group('s3user', 'S3Admins')

###################### DynamoDB ######################

'''
PartiQL
query 1
select * from "dynamodb-testtable" where id='1'

query 2
insert into "dynamodb-testtable" value {
	'id':'3',
    'date':'2020-04-11',
    'summary':'Scrum methodology to implement Agile philosophy for Analytics teams.'
}

query 3
delete from "dynamodb-testtable" where id='1' and "date"='2020-02-15'

query - scan operation vs global secondary index
scan + FilterExpression()
'''

# insert data using resource
# put_item write 1 item at a time
db = boto3.resource('dynamodb')
table = db.Table('dynamodb-testtable')
table.put_item(
    Item = {
        'id':"4",
        'date':"2020-05-09",
        'summary':"A three part article series on version control using Git and GitHub."
    }
)

# inset data using client
db = boto3.client('dynamodb')
response = db.put_item(
    TableName = 'dynamodb-testtable',
    Item = {
        'id':{'S':"5"}, # S is for string
        'date':{'S':"2020-08-01"},
        'summary':{"S":"What is Markdown?"}
    }
)

# batch_writer can write upto 25 items at a time
db = boto3.resource('dynamodb')
table = db.Table('dynamodb-testtable')
with table.batch_writer() as batch:
    batch.put_item(
        Item = {
            'id':"6",
            'date':"2020-08-08",
            'summary':"Learn how to automate repetitive or complex tasks using the power of Excel VBA."
        }
    )
    batch.put_item(
        Item = {
            'id':"7",
            'date':"2020-08-15",
            'summary':"C++ is between 10 and 100 times faster than Python when doing any serious number crunching."
        }
    )
    batch.put_item(
        Item = {
            'id':"8",
            'date':"2020-09-05",
            'summary':"C++ is between 10 and 100 times faster than Python when doing any serious number crunching."
        }
    )

# describe table
db = boto3.client('dynamodb')
response = db.describe_table(
    TableName = 'dynamodb-testtable'
)
from pprint import pprint
pprint(response)

# list tables
db = boto3.client('dynamodb')
response = db.list_tables()
print(response['TableNames'])

# update/change table configuration
db = boto3.client('dynamodb')
response = db.update_table(
    TableName='dynamodb-testtable',
    BillingMode='PROVISIONED',
    ProvisionedThroughput={
        'ReadCapacityUnits':1,
        'WriteCapacityUnits':1
    }
)
print(response)

# creating backup
db = boto3.client('dynamodb')
response = db.create_backup(
    TableName='dynamodb-testtable',
    BackupName='dynamodb-testtable-backup'
)
print(response)

# deleting backup
db = boto3.client('dynamodb')
response = db.delete_backup(
    BackupArn='arn:aws:dynamodb:us-east-1:001815726351:table/dynamodb-testtable/backup/01656104139211-42159204'
)
print(response)

# get item from dynamodb - resource vs client method
# retrieves single item using primary key
# resource method
db = boto3.resource('dynamodb')
table = db.Table('dynamodb-testtable')
response = table.get_item(
    Key = {
        'id':"7",
        'date':"2020-08-15"
    }
)
print(response['Item'])

# client method
db = boto3.client('dynamodb')
response = db.get_item(
    TableName='dynamodb-testtable',
    Key={
        'id':{
            'S':'1'
        },
        'date':{
            'S':"2020-02-15"
        }
    }
)
print(response['Item'])

# get batch item - retrieves upto 100 items in one network call
db = boto3.resource('dynamodb')
response = db.batch_get_item(
    RequestItems={
        'dynamodb-testtable':{
            'Keys':[
                {
                    'id':'1', 'date':"2020-02-15"
                },
                {
                    'id':'2', 'date':"2020-02-15"
                }
            ]
        }
    }
)
from pprint import pprint
pprint(response['Responses'])

# scan through dynamodb table
# scan has 1mb limit in amount of data it returns
# we have topaginate through results using a loop
from pprint import pprint
db = boto3.resource('dynamodb')
table = db.Table('dynamodb-testtable')
response = table.scan()
data = response['Items']
pprint(data)

# create a table
def create_movie_table(dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.create_table(
        TableName='Movies',
        KeySchema=[
            {
                'AttributeName':'year',
                'KeyType':'HASH'
            },
            {
                'AttributeName': 'title',
                'KeyType': 'RANGE'
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'year',
                'AttributeType': 'N'
            },
            {
                'AttributeName': 'title',
                'AttributeType': 'S'
            }
        ],
        ProvisionedThroughput = {
            'ReadCapacityUnits':10,
            'WriteCapacityUnits':10
        }
    )
    return table
if __name__ == "__main__":
    movie_table = create_movie_table()
    print("Table status : ", movie_table.table_status)

# load json file into dynamodb table
from decimal import Decimal
def load_movie(movies, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Movies')
    for movie in movies:
        year = int(movie['year'])
        title = movie['title']
        print("Adding movie : ", year, title)
        table.put_item(Item=movie)
if __name__=="__main__":
    with open('moviedata.json') as json_file:
        movie_list = json.load(json_file, parse_float=Decimal)
    load_movie(movie_list)

# get movies data from dynamodb table
from pprint import pprint
from botocore.exceptions import ClientError
def get_movie(title, year, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Movies')
    try:
        response = table.get_item(Key={'year':year, 'title':title})
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        return response['Item']
if __name__ == "__main__":
    movie = get_movie('The Boondock Saints', 1999)
    if movie:
        pprint(movie)

# updating the data
from pprint import pprint
from decimal import Decimal
def update_movie(title, year, rating, plot, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Movies')
    response = table.update_item(
        Key={
            'year':year,
            'title':title
        },
        UpdateExpression="set info.rating=:r, info.plot=:p",
        ExpressionAttributeValues = {
            ':r': Decimal(rating),
            ':p': plot,
        },
        ReturnValues='UPDATED_NEW'
    )
    return response
if __name__ == "__main__":
    update_response = update_movie(
        "The Shawshank Redemption", 1994, "5.4", "This is just for testing"
    )
    pprint(update_response)

# delete a movie from the dynamodb table
from pprint import pprint
from botocore.exceptions import ClientError
def delete_movie(title, year, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Movies')
    try:
        response = table.delete_item(
            Key={
                'year':year,
                'title':title
            }
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        return response
if __name__ == "__main__":
    delete_response = delete_movie("Now You See Me", 2013)
    if delete_response:
        pprint(delete_response)

# get all movies
from boto3.dynamodb.conditions import Key
def query_movies(year, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Movies')
    response = table.query(
        KeyConditionExpression=Key('year').eq(year)
    )
    return response['Items']
if __name__ == "__main__":
    query_year=2013
    print("Movies from {} ".format(query_year))
    movies = query_movies(query_year)
    for movie in movies:
        print(movie['year'], ":", movie['title'])

######################### s3 #########################

# creating bucket using resources
bucket = boto3.resource('s3')
response = bucket.create_bucket(
    Bucket = "vivek20220706",
    ACL="private", # public-read, public-read-write
)
print(response)

# create bucket using client method
client = boto3.client('s3')
response = client.create_bucket(
    Bucket = "vivek20220706b",
    ACL = "private",
)
print(response)

# Listing all buckets - using client
bucket = boto3.client('s3')
response = bucket.list_buckets()
print("Listing all buckets")
for bucket in response['Buckets']:
    print(bucket['Name'])

# Listing all buckets - using resource
resource = boto3.resource('s3')
iterator = resource.buckets.all()
print("Listing all buckets")
for bucket in iterator:
    print(bucket.name)

# Delete bucket - using client
client = boto3.client('s3')
bucket_name = "vivek20220706"
client.delete_bucket(Bucket=bucket_name)
print("S3 Bucket has been deleted")

# Delete bucket - using resource
resource = boto3.resource('s3')
bucket_name = "vivek20220706"
s3_bucket =resource.Bucket(bucket_name)
s3_bucket.delete()
print(" This {} bucket has been deleted  ".format(s3_bucket))

# Delete non-empty bucket - using resource
BUCKET_NAME = "vivek20220706"
s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket(BUCKET_NAME)
def clean_up():
    #delete the object
    for s3_object in s3_bucket.objects.all():
        s3_object.delete()
    #delete bucket versioning
    for s3_object_ver in s3_bucket.object_versions.all():
        s3_object_ver.delete()
    print("S3 bucket cleaned")
clean_up()
s3_bucket.delete()
print("The bucket has been deleted")

# uploading file to bucket - using client - you can specify custom key (filename in aws)
client = boto3.client('s3')
with open(r'C:\Users\vivek\Documents\Code\Applying-Data-Science\vp-test.txt', 'rb') as f:
    data = f.read()
response = client.put_object(
    ACL="private", #"public-read-write",
    Bucket = "vivek20220706",
    Body=data,
    Key='vp-test.txt'
)
print(response)

# Upload file - using client (alternate way)
s3_client = boto3.client('s3')
def upload_files(file_name, bucket, object_name=None, args=None):
    if object_name is None:
        object_name = file_name
    s3_client.upload_file(file_name, bucket, object_name, ExtraArgs=args)
    print("{} has been uploaded to {} bucket".format(file_name, BUCKET_NAME))
upload_files(r'C:\Users\vivek\Documents\Code\Applying-Data-Science\vp-test-b.txt', 'vivek20220706')

# Upload file - using resource
s3_client = boto3.resource('s3')
def upload_files(file_name, bucket, object_name=None, args=None):
    if object_name is None:
        object_name = file_name
    s3_client.meta.client.upload_file(file_name, bucket, object_name, ExtraArgs=args)
    print("{} has been uploaded to {} bucket".format(file_name, BUCKET_NAME))
upload_files(r'C:\Users\vivek\Documents\Code\Applying-Data-Science\vp-test-c.txt', 'vivek20220706')

# Download file
BUCKET_NAME = "vivek20220706"
s3_resource = boto3.resource('s3')
s3_object = s3_resource.Object(BUCKET_NAME, 'vp-test.txt')
s3_object.download_file('vp-test.txt')
print("File has been downloaded")

# Listing Bucket Files or objects
BUCKET_NAME = "vivek20220706"
s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket(BUCKET_NAME)
print("Listing Bucket Files or Objects")
for obj in s3_bucket.objects.all():
    print(obj.key)

# Listing specific files from a bucket
BUCKET_NAME = "vivek20220706"
s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket(BUCKET_NAME)
print("Listing Filtered File")
for obj in s3_bucket.objects.filter(Prefix="vp-"):
    print(obj.key)

# Getting summary of an object - bucket name and object key
s3 = boto3.resource('s3')
object_summary = s3.ObjectSummary("vivek20220706", "vp-test.txt")
print(object_summary.bucket_name)
print(object_summary.key)

# Create a copy of a file
s3 = boto3.resource('s3')
copy_source = {
    'Bucket':'vivek20220706',
    'Key':'vp-test.txt'
}
s3.meta.client.copy(copy_source, 'vivek20220706', 'vp-test-copy.txt')

# delete file from bucket - using client
client = boto3.client('s3')
response = client.delete_object(
    Bucket = 'vivek20220706',
    Key='vp-test-copy-2.txt'
)
print(response)

# delete multiple files or objects
response = client.delete_objects(
    Bucket = 'vivek20220706',
    Delete = {
        'Objects':[ 
            {'Key':'vp-test-copy-2.txt'},
            {'Key':'vp-test-copy-3.txt'}
        ]
    }
)
print(response)

# HOSTING STATIC WEBSITE IN S3
# Install Node.js
# Create a reactjs app - https://reactjs.org/docs/create-a-new-react-app.html
npx create-react-app my-app
cd my-app
npm start

# Create an aws bucket (uncheck block public access, so bucket/website can be accssed publically)

# Build reactjs app - this will create build folder in my-app folder
npm run build

# Upload all files in build folder to the s3 bucket
# Upload any subfolders in build folder to s3 bucket, one-by-one

# enable static website hosting in bucket properties 
# add index.html as name of index page
# after this you will see the link for the static website in bucket properties/static website hosting section

# update bucket policy
# go to permissions > bucket policy section > edit > click on policy generator
# type of policy - s3 bucket policy, effect - allow, principal - *, action - GetObject, arn/* - get arn from bucket properties, add /* in the end
# copy paste this generated policy in permission > bucket policy section

# Get website configuration from a bucket
from pprint import pprint
client = boto3.client('s3')
response = client.get_bucket_website(
    Bucket="awstutorial12"
)
pprint(response)

# Delete attached policy from bukcet
client = boto3.client('s3')
response = client.delete_bucket_policy(
    Bucket="awstutorial12"
)
print(response)

# Delete website from bucket
client = boto3.client('s3')
response = client.delete_bucket_website(
    Bucket="awstutorial12"
)
print(response)

# Lambda funcitons - serverless computing services to run code
# you only pay for time when your code is running
# several options to create a lambda function - from scratch, using a blueprint, container image, from serverless app repository
# specify - function name, runtime - python 3.9, architecture - x86_64, execution role - create a new role with basic lambda permissions > create
# a template is created with filename lambda_function.py with a function lambda_handler(event, context)
# configure event by clicking on Test > event name - MyFirestEvent > save
# run by clicking on Test (again)


# LAMBDA
# Dummy Labda code - lambda_function.py
import json
def lambda_handler(event, context):
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

# listing s3 buckets with lambda function
# we will need to create a policy that allows lambda function access to list s3 bukets
# IAM > Role > Create role > choose role AmazonS3FullAccess
import json
import boto3  # python sdk and it is available in here you don't need t install
s3 = boto3.resource('s3')
def lambda_handler(event, context):
    s3_buckets = []
    for bucket in s3.buckets.all():
        s3_buckets.append(bucket.name)
    return {
        "statusCode": 200,
        "body": s3_buckets
    }


s3://

