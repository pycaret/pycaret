import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


def create_container(container_name):

  # Create the container
  container_client = blob_service_client.create_container(container_name)

  return container_client

def upload_blob(container_name, source_file_name, destination_blob_name):

  # Create a blob client using the local file name as the name for the blob
  blob_client = blob_service_client.get_blob_client(container=container_name, blob=destination_blob_name)

  print("\nUploading to Azure Storage as blob:\n\t" + source_file_name)

  # Upload the created file
  with open(source_file_name, "rb") as data:
      blob_client.upload_blob(data)


def download_blob(container_name, source_blob_name, destination_file_name):
  # Download the blob to a local file
  print("\nDownloading blob to \n\t" + destination_file_name)

  # Create a blob client using the local file name as the name for the blob
  blob_client = blob_service_client.get_blob_client(container=container_name, blob=source_blob_name)

  if destination_file_name is not None:
        with open(destination_file_name, "wb") as download_file:
          download_file.write(blob_client.download_blob().readall())

        print(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_file_name
            )
        )