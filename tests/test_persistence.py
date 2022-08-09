import os

import boto3
import pytest
from moto import mock_s3

from pycaret.internal.persistence import deploy_model, load_model


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture(scope="function")
def s3(aws_credentials):
    """Create a mock s3 for testing."""
    with mock_s3():
        yield boto3.client("s3", region_name="us-east-1")


def test_deploy_model(s3):
    authentication = {"bucket": "pycaret-test", "path": "test"}
    s3.create_bucket(Bucket=authentication.get("bucket"))

    model = "test_model"
    model_name = "test"

    deploy_model(
        model, model_name=model_name, platform="aws", authentication=authentication
    )

    s3.head_object(
        Bucket=authentication.get("bucket"),
        Key=os.path.join(authentication.get("path"), f"{model_name}.pkl"),
    )

    _ = load_model(
        model_name=model_name,
        platform="aws",
        authentication=authentication,
        verbose=True,
    )
