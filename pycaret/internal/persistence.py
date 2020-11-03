# Module: internal.persistence
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Dict, Optional
from pycaret.internal.utils import get_logger
from pycaret.internal.Display import Display
from sklearn.pipeline import Pipeline
import gc


def deploy_model(
    model, model_name: str, authentication: dict, platform: str = "aws", prep_pipe_=None
):

    """
    (In Preview)

    This generic function deploys the transformation pipeline and trained model object for
    production use. The platform of deployment can be defined under the platform
    param along with the applicable authentication tokens which are passed as a
    dictionary to the authentication param.
    
    Notes
    -----
    For AWS users:
    Before deploying a model to an AWS S3 ('aws'), environment variables must be 
    configured using the command line interface. To configure AWS env. variables, 
    type aws configure in your python command line. The following information is
    required which can be generated using the Identity and Access Management (IAM) 
    portal of your amazon console account:

    - AWS Access Key ID
    - AWS Secret Key Access
    - Default Region Name (can be seen under Global settings on your AWS console)
    - Default output format (must be left blank)

    For GCP users:
    --------------
    Before deploying a model to Google Cloud Platform (GCP), user has to create Project
    on the platform from consol. To do that, user must have google cloud account or
    create new one. After creating a service account, down the JSON authetication file
    and configure  GOOGLE_APPLICATION_CREDENTIALS= <path-to-json> from command line. If
    using google-colab then authetication can be done using `google.colab` auth method.
    Read below link for more details.

    https://cloud.google.com/docs/authentication/production

    - Google Cloud Project
    - Service Account Authetication

    For Azure users:
    ---------------
    Before deploying a model to Microsoft's Azure (Azure), environment variables
    for connection string must be set. In order to get connection string, user has
    to create account of Azure. Once it is done, create a Storage account. In the settings
    section of storage account, user can get the connection string.

    Read below link for more details.
    https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json

    - Azure Storage Account

    Parameters
    ----------
    model : object
        A trained model object should be passed as an estimator. 
    
    model_name : str
        Name of model to be passed as a string.
    
    authentication : dict
        Dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'Name of Bucket on S3'}

        When platform = 'gcp':
        {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

        When platform = 'azure':
        {'container': 'pycaret-test'}
    
    platform: str, default = 'aws'
        Name of platform for deployment. Current available options are: 'aws', 'gcp' and 'azure'

    Returns
    -------
    Success_Message
    
    Warnings
    --------
    - This function uses file storage services to deploy the model on cloud platform. 
      As such, this is efficient for batch-use. Where the production objective is to 
      obtain prediction at an instance level, this may not be the efficient choice as 
      it transmits the binary pickle file between your local python environment and
      the platform. 
    
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing deploy_model()")
    logger.info(f"deploy_model({function_params_str})")

    import sys

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    allowed_platforms = ["aws", "gcp", "azure"]

    if platform not in allowed_platforms:
        logger.error(
            f"(Value Error): Platform {platform} is not supported by pycaret or illegal option"
        )
        raise ValueError(
            f"Platform {platform} is not supported by pycaret or illegal option"
        )

    if platform:
        if not authentication:
            raise ValueError("Authentication is missing.")

    # general dependencies
    import ipywidgets as ipw
    import pandas as pd
    from IPython.display import clear_output, update_display
    import os

    logger.info("Saving model in active working directory")
    logger.info("SubProcess save_model() called ==================================")
    save_model(model, prep_pipe_=prep_pipe_, model_name=model_name, verbose=False)
    logger.info("SubProcess save_model() end ==================================")

    if platform == "aws":

        logger.info("Platform : AWS S3")

        # checking if awscli available
        try:
            import awscli
        except:
            logger.error(
                "awscli library not found. pip install awscli to use deploy_model function."
            )
            raise ImportError(
                "awscli library not found. pip install awscli to use deploy_model function."
            )

        import boto3

        # initiaze s3
        logger.info("Initializing S3 client")
        s3 = boto3.client("s3")
        filename = f"{model_name}.pkl"
        key = f"{model_name}.pkl"
        bucket_name = authentication.get("bucket")
        s3.upload_file(filename, bucket_name, key)
        clear_output()
        os.remove(filename)
        print("Model Succesfully Deployed on AWS S3")
        logger.info("Model Succesfully Deployed on AWS S3")
        logger.info(str(model))

    elif platform == "gcp":

        logger.info("Platform : GCP")

        try:
            import google.cloud
        except:
            logger.error(
                "google-cloud-storage library not found. pip install google-cloud-storage to use deploy_model function with GCP."
            )
            raise ImportError(
                "google-cloud-storage library not found. pip install google-cloud-storage to use deploy_model function with GCP."
            )

        # initialize deployment
        filename = f"{model_name}.pkl"
        key = f"{model_name}.pkl"
        bucket_name = authentication.get("bucket")
        project_name = authentication.get("project")
        try:
            _create_bucket_gcp(project_name, bucket_name)
            _upload_blob_gcp(project_name, bucket_name, filename, key)
        except:
            _upload_blob_gcp(project_name, bucket_name, filename, key)
        os.remove(filename)
        print("Model Succesfully Deployed on GCP")
        logger.info("Model Succesfully Deployed on GCP")
        logger.info(str(model))

    elif platform == "azure":

        try:
            import azure.storage.blob
        except:
            logger.error(
                "azure-storage-blob library not found. pip install azure-storage-blob to use deploy_model function with Azure."
            )
            raise ImportError(
                "azure-storage-blob library not found. pip install azure-storage-blob to use deploy_model function with Azure."
            )

        logger.info("Platform : Azure Blob Storage")

        # initialize deployment
        filename = f"{model_name}.pkl"
        key = f"{model_name}.pkl"
        container_name = authentication.get("container")
        try:
            container_client = _create_container_azure(container_name)
            _upload_blob_azure(container_name, filename, key)
            del container_client
        except:
            _upload_blob_azure(container_name, filename, key)

        os.remove(filename)

        print("Model Succesfully Deployed on Azure Storage Blob")
        logger.info("Model Succesfully Deployed on Azure Storage Blob")
        logger.info(str(model))

    logger.info(
        "deploy_model() succesfully completed......................................"
    )
    gc.collect()


def save_model(model, model_name: str, prep_pipe_=None, verbose: bool = True):

    """
    This generic function saves the transformation pipeline and trained model object 
    into the current active directory as a pickle file for later use. 

    Parameters
    ----------
    model : object
        A trained model object should be passed as an estimator. 
    
    model_name : str
        Name of pickle file to be passed as a string.
    
    prep_pipe_ : Pipeline, default = None
        If not None, will save the entire Pipeline in addition to model.

    verbose: bool, default = True
        Success message is not printed when verbose is set to False.

    Returns
    -------
    (model, model_filename):
        Tuple of the model object and the filename it was saved under.

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing save_model()")
    logger.info(f"save_model({function_params_str})")

    from copy import deepcopy

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    logger.info("Adding model into prep_pipe")

    if isinstance(model, Pipeline):
        model_ = deepcopy(model)
        logger.warning("Only Model saved as it was a pipeline.")
    elif not prep_pipe_:
        model_ = deepcopy(model)
        logger.warning("Only Model saved. Transformations in prep_pipe are ignored.")
    else:
        model_ = deepcopy(prep_pipe_)
        model_.steps.append(["trained_model", model])

    import joblib

    model_name = f"{model_name}.pkl"
    joblib.dump(model_, model_name)
    if verbose:
        print("Transformation Pipeline and Model Succesfully Saved")

    logger.info(f"{model_name} saved in current working directory")
    logger.info(str(model_))
    logger.info(
        "save_model() succesfully completed......................................"
    )
    gc.collect()
    return (model_, model_name)


def load_model(
    model_name,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):

    """
    This generic function loads a previously saved transformation pipeline and model 
    from the current active directory into the current python environment. 
    Load object must be a pickle file.

    Parameters
    ----------
    model_name : str, default = none
        Name of pickle file to be passed as a string.
      
    platform: str, default = None
        Name of platform, if loading model from cloud. Current available options are:
        'aws', 'gcp' and 'azure'.
    
    authentication : dict
        dictionary of applicable authentication tokens.

        When platform = 'aws':
        {'bucket' : 'Name of Bucket on S3'}

        When platform = 'gcp':
        {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

        When platform = 'azure':
        {'container': 'pycaret-test'}
    
    verbose: bool, default = True
        Success message is not printed when verbose is set to False.

    Returns
    -------
    Model Object

    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing load_model()")
    logger.info(f"load_model({function_params_str})")

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # exception checking
    import sys

    if platform:
        if not authentication:
            raise ValueError("Authentication is missing.")

    if not platform:

        import joblib

        model_name = f"{model_name}.pkl"
        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")
        return joblib.load(model_name)

    # cloud providers
    elif platform == "aws":

        import boto3

        bucketname = authentication.get("bucket")
        filename = f"{model_name}.pkl"
        s3 = boto3.resource("s3")
        s3.Bucket(bucketname).download_file(filename, filename)
        filename = str(model_name)
        model = load_model(filename, verbose=False)
        model = load_model(filename, verbose=False)

        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")

        return model

    elif platform == "gcp":

        bucket_name = authentication.get("bucket")
        project_name = authentication.get("project")
        filename = f"{model_name}.pkl"

        model_downloaded = _download_blob_gcp(
            project_name, bucket_name, filename, filename
        )

        model = load_model(model_name, verbose=False)

        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")
        return model

    elif platform == "azure":

        container_name = authentication.get("container")
        filename = f"{model_name}.pkl"

        model_downloaded = _download_blob_azure(container_name, filename, filename)

        model = load_model(model_name, verbose=False)

        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")
        return model
    else:
        print(f"Platform {platform} is not supported by pycaret or illegal option")
    gc.collect()


def _create_bucket_gcp(project_name: str, bucket_name: str):
    """
    Creates a bucket on Google Cloud Platform if it does not exists already

    Example
    -------
    >>> _create_bucket_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp')

    Parameters
    ----------
    project_name : str
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : str
        Name of the storage bucket to be created if does not exists already.

    Returns
    -------
    None
    """

    logger = get_logger()

    # bucket_name = "your-new-bucket-name"
    from google.cloud import storage

    storage_client = storage.Client(project_name)

    buckets = storage_client.list_buckets()

    if bucket_name not in buckets:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info("Bucket {} created".format(bucket.name))
    else:
        raise FileExistsError("{} already exists".format(bucket_name))


def _upload_blob_gcp(
    project_name: str,
    bucket_name: str,
    source_file_name: str,
    destination_blob_name: str,
):

    """
    Upload blob to GCP storage bucket

    Example
    -------
    >>> _upload_blob_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp', \
                        source_file_name='model-101.pkl', destination_blob_name='model-101.pkl')

    Parameters
    ----------
    project_name : str
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : str
        Name of the storage bucket to be created if does not exists already.

    source_file_name : str
        A blob/file name to copy to GCP

    destination_blob_name : str
        Name of the destination file to be stored on GCP

    Returns
    -------
    None
    """

    logger = get_logger()

    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    from google.cloud import storage

    storage_client = storage.Client(project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info(
        "File {} uploaded to {}.".format(source_file_name, destination_blob_name)
    )


def _download_blob_gcp(
    project_name: str,
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str,
):
    """
    Download a blob from GCP storage bucket

    Example
    -------
    >>> _download_blob_gcp(project_name='GCP-Essentials', bucket_name='test-pycaret-gcp', \
                          source_blob_name='model-101.pkl', destination_file_name='model-101.pkl')

    Parameters
    ----------
    project_name : str
        A Project name on GCP Platform (Must have been created from console).

    bucket_name : str
        Name of the storage bucket to be created if does not exists already.

    source_blob_name : str
        A blob/file name to download from GCP bucket

    destination_file_name : str
        Name of the destination file to be stored locally

    Returns
    -------
    Model Object
    """

    logger = get_logger()

    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    from google.cloud import storage

    storage_client = storage.Client(project_name)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if destination_file_name is not None:
        blob.download_to_filename(destination_file_name)

        logger.info(
            "Blob {} downloaded to {}.".format(source_blob_name, destination_file_name)
        )

    return blob


def _create_container_azure(container_name: str):
    """
    Creates a storage container on Azure Platform. gets the connection string from the environment variables.

    Example
    -------
    >>>  container_client = _create_container_azure(container_name='test-pycaret-azure')

    Parameters
    ----------
    container_name : str
        Name of the storage container to be created if does not exists already.

    Returns
    -------
    cotainer_client

    """

    logger = get_logger()

    # Create the container
    import os, uuid
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.create_container(container_name)
    return container_client


def _upload_blob_azure(
    container_name: str, source_file_name: str, destination_blob_name: str
):
    """
    Upload blob to Azure storage  container

    Example
    -------
    >>>  _upload_blob_azure(container_name='test-pycaret-azure', source_file_name='model-101.pkl', \
                           destination_blob_name='model-101.pkl')

    Parameters
    ----------
    container_name : str
        Name of the storage bucket to be created if does not exists already.

    source_file_name : str
        A blob/file name to copy to Azure

    destination_blob_name : str
        Name of the destination file to be stored on Azure

    """

    logger = get_logger()

    import os, uuid
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=destination_blob_name
    )

    # Upload the created file
    with open(source_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)


def _download_blob_azure(
    container_name: str, source_blob_name: str, destination_file_name: str
):
    """
    Download blob from Azure storage  container

    Example
    -------
    >>>  _download_blob_azure(container_name='test-pycaret-azure', source_blob_name='model-101.pkl', \
                             destination_file_name='model-101.pkl')

    Parameters
    ----------
    container_name : str
        Name of the storage bucket to be created if does not exists already.

    source_blob_name : str
        A blob/file name to download from Azure storage container

    destination_file_name : str
        Name of the destination file to be stored locally

    """

    logger = get_logger()

    import os, uuid
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=source_blob_name
    )

    if destination_file_name is not None:
        with open(destination_file_name, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
