from typing import List, Optional
import os
import subprocess
import logging
from datetime import datetime

from awscliv2.installers import install_linux

LINUX_X86_64_URL = "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"

install_linux(LINUX_X86_64_URL)

logger = logging.getLogger(__name__)


def get_checkpoint_and_refs_dir(
    model_id: str,
    bucket_uri: str,
    s3_sync_args: Optional[List[str]] = None,
    mkdir: bool = False,
) -> str:

    from transformers.utils.hub import TRANSFORMERS_CACHE

    f_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")

    refs_dir = os.path.join(path, "refs")
    checkpoint_dir = os.path.join(path, "snapshots", f_timestamp)

    if mkdir:
        os.makedirs(refs_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir, refs_dir


def get_download_path(model_id: str):
    from transformers.utils.hub import TRANSFORMERS_CACHE

    path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
    return path


def download_model(
    model_id: str,
    bucket_uri: str,
    s3_sync_args: Optional[List[str]] = None,
    tokenizer_only: bool = False,
) -> None:
    """
    Download a model from an S3 bucket and save it in TRANSFORMERS_CACHE for
    seamless interoperability with Hugging Face's Transformers library.
    """
    s3_sync_args = s3_sync_args or []
    path = get_download_path(model_id)

    cmd = (
        ["awsv2", "s3", "sync"]
        + s3_sync_args
        + (["--exclude", "*", "--include", "*token*"] if tokenizer_only else [])
        + [bucket_uri, path]
    )
    print(f"RUN({cmd})")
    subprocess.run(cmd)
    print("done")


def get_mirror_link(model_id: str) -> str:
    return f"s3://llama-2-weights/models--{model_id.replace('/', '--')}"