import boto3

client = boto3.resource('s3')


'''
response = client.delete_object(
    Bucket = 'parwizforogh7777',
    Key='file.pdf'
)

print(response)

'''


#delete multiple files or objects

response = client.delete_objects(
    Bucket = 'parwizforogh7777',
    Delete = {
        'Objects':[
            {
                'Key':'file.txt'
            },

            {
                'Key':'myfile.txt'
            }


        ]
    }
)

print(response)


