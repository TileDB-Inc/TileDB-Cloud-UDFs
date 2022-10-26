import tiledb
import tiledb.cloud
import public_udfs

from contextlib import suppress
import numpy as np
from numpy.testing import assert_array_equal
import os
import pandas as pd
import platform
import pytest
import time
import types


@pytest.fixture(autouse=True, scope="session")
def tiledb_cloud_login():
    """
    The TILEDB_REST_TOKEN for accessing the unittest TileDB Cloud namespace must
    be set by the user.
    """
    tiledb.cloud.login(token=os.environ["TILEDB_REST_TOKEN"])


@pytest.fixture(scope="session")
def key():
    """
    The AWS_ACCESS_KEY_ID environment variable for accessing the unittest
    S3 Bucket must be set by the user.
    """
    return os.environ["AWS_ACCESS_KEY_ID"]


@pytest.fixture(scope="session")
def secret():
    """
    The AWS_SECRET_ACCESS_KEY environment variable for accessing the unittest
    S3 Bucket must be set by the user.
    """
    return os.environ["AWS_SECRET_ACCESS_KEY"]


@pytest.fixture(scope="session")
def namespace():
    return "unittest"


@pytest.fixture(scope="session")
def bucket():
    return "tiledb-unittest"


@pytest.fixture(scope="function")
def array_name(request):
    return f"{request.param}-{platform.system()}-{platform.release()}-python{platform.python_version()}-{int(time.time())}"


@pytest.fixture(scope="function")
def udf_uri(request, namespace):
    """
    Register or update the UDF, given by request, for the given namespace.
    """
    # Pass empty globals dictionary. Functions defined outside of __main__
    # are not supported by Cloudpickle, but TileDB public UDFs are defined in
    # public_udfs.py. This workaround allows the UDF to be unpickled by
    # striping the public_udf module.
    # https://stackoverflow.com/questions/49821323/python3-pickle-a-function-without-side-effects
    udf = types.FunctionType(request.param.__code__, {})
    test_udf_name = "test_{}".format(udf.__name__)

    udf_names = [
        arr.name
        for arr in tiledb.cloud.client.list_arrays(
            file_type=[tiledb.cloud.rest_api.models.FileType.USER_DEFINED_FUNCTION],
            namespace=namespace,
        ).arrays
    ]

    if test_udf_name not in udf_names:
        tiledb.cloud.udf.register_generic_udf(udf, test_udf_name)
    else:
        tiledb.cloud.udf.update_generic_udf(udf, test_udf_name)

    yield f"{namespace}/{test_udf_name}"

    # TODO should delete the test UDF at teardown.
    # tiledb.cloud.client.client.udf_api.delete_udf_info(namespace, test_udf_name)


@pytest.fixture(scope="session")
def config():
    """
    The TILEDB_REST_TOKEN for accessing the unittest TileDB Cloud namespace must
    be set by the user.
    """
    config = tiledb.Config()
    config["rest.token"] = os.environ["TILEDB_REST_TOKEN"]
    return config


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_vcf_samples, "test_ingest_vcf_samples")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_vcf_samples(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    def set_up(key, secret, bucket, array_name):
        import tiledbvcf

        print(f"TileDB-VCF version: {tiledbvcf.version}")

        ds = tiledbvcf.Dataset(
            f"s3://{bucket}/{array_name}.tdb",
            mode="w",
            cfg=tiledbvcf.ReadConfig(
                tiledb_config={
                    "vfs.s3.aws_access_key_id": key,
                    "vfs.s3.aws_secret_access_key": secret,
                    "vfs.s3.connect_timeout_ms": 120000,
                    "vfs.s3.request_timeout_ms": 120000,
                }
            ),
        )
        ds.create_dataset(
            vcf_attrs="s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00096/analysis/HG00096.haplotypeCalls.er.raw.vcf.gz"
        )

    tiledb.cloud.udf.exec(set_up, key, secret, bucket, array_name)

    time.sleep(10)

    tiledb.cloud.udf.exec(
        f"s3://{bucket}/{array_name}.tdb",
        "chr1",
        [
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00096/analysis/HG00096.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00097/analysis/HG00097.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00099/analysis/HG00099.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00100/analysis/HG00100.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00101/analysis/HG00101.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00102/analysis/HG00102.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00103/analysis/HG00103.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00105/analysis/HG00105.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00106/analysis/HG00106.haplotypeCalls.er.raw.vcf.gz",
            "s3://genomic-datasets/vcf/1kg/1kghicov/vcfs/GBR/Sample_HG00107/analysis/HG00107.haplotypeCalls.er.raw.vcf.gz",
        ],
        (0, 2),
        {
            "vfs.s3.aws_access_key_id": key,
            "vfs.s3.aws_secret_access_key": secret,
            "vfs.s3.connect_timeout_ms": 120000,
            "vfs.s3.request_timeout_ms": 120000,
        },
        7168,
        4,
        True,
        True,
        image_name="genomics",
        name=udf_uri,
    )

    print("done executing", file=sys.stderr)

    time.sleep(10)

    def test_vcf_dataset(key, secret, bucket, array_name):
        import tiledbvcf

        ds = tiledbvcf.Dataset(
            f"s3://{bucket}/{array_name}.tdb",
            mode="r",
            cfg=tiledbvcf.ReadConfig(
                tiledb_config={
                    "vfs.s3.aws_access_key_id": key,
                    "vfs.s3.aws_secret_access_key": secret,
                    "vfs.s3.connect_timeout_ms": 120000,
                    "vfs.s3.request_timeout_ms": 120000,
                }
            ),
        )

    tiledb.cloud.udf.exec(test_vcf_dataset, key, secret, bucket, array_name)


# This will likely be moved into its own class in the future. The class will
# group UDFs that write arrays to TileDB Cloud.
@pytest.fixture(autouse=True, scope="function")
def clean_arrays(array_name, namespace, bucket):
    """
    Remove the given array_name from the S3 Bucket located at bucket and TileDB
    Cloud located at namespace. This is a set up and teardown function that runs
    for all unit tests. This runs at the beginning of all tests, prior to
    calling the UDF, to ensure that the array does not exist prior to writing.
    It also runs at  end of all tests, regardless or passing, failing, or
    prematurely erroring out, to remove the array.
    """
    tiledb_uri = f"tiledb://{namespace}/{array_name}.tdb"
    s3_uri = f"s3://{bucket}/{array_name}.tdb"

    yield

    # Supressing errors is a temporary solution to delays between writing (or in
    # this case, deleting) and reading arrays on S3.

    with suppress(Exception):
        tiledb.cloud.deregister_array(tiledb_uri)

    with suppress(Exception):
        tiledb.remove(s3_uri)


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "test_ingest_csv_sparse_array")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_csv_sparse_array(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a sparse array from a CSV file using ingest_csv().
    """
    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        name=udf_uri,  # unittest/test_ingest_csv --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        data = pd.DataFrame(A[:])

        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                data[attribute],
                np.array([row * 10 + col for row in range(1, 21)]),
            )


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "test_ingest_csv_sparse_array_apppend")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_csv_sparse_array_apppend(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a sparse array from a CSV file using ingest_csv() in the default
    ingest mode and then append additional data to it using the append mode.
    """
    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment_sparse1.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        mode="ingest",
        full_domain=True,
        index_col=("x"),
        sparse=True,
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        assert A.nonempty_domain() == ((1, 20),)

    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment_sparse2.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        mode="append",
        full_domain=True,
        index_col=("x"),
        row_start_idx=20,
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                A.df[:][attribute],
                np.array([row * 10 + col for row in range(1, 21)] * 2),
            )


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "test_ingest_csv_sparse_array_apppend_header_mismatch")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_csv_sparse_array_apppend_header_mismatch(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a sparse array from a CSV file using ingest_csv() in the default
    ingest mode and then append additional data to it using the append mode.
    The appended data contains header names that do not match the data in the
    sparse array and must be renamed.
    """
    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment_sparse1.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        mode="ingest",
        full_domain=True,
        index_col=("x"),
        sparse=True,
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        assert A.nonempty_domain() == ((1, 20),)

    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment_sparse2_mismatch.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        mode="append",
        full_domain=True,
        index_col=("x"),
        header=0,
        names=["x", "c", "b", "a"],
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                A.df[:][attribute],
                np.array([row * 10 + col for row in range(1, 21)] * 2),
            )


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "test_ingest_csv_sparse_array_null_replace")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_csv_sparse_array_null_replace(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    From a CSV file containing NaNs, produce a sparse array using ingest_csv()
    where the NaNs are replaced with the value given by fillna.
    """
    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment_nulls.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        fillna={"b": 321, "c": 123},
        sparse=True,
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        assert_array_equal(A.df[:]["a"], np.array([1, 1, 1]))
        assert_array_equal(A.df[:]["b"], np.array([2, 2, 321]))
        assert_array_equal(A.df[:]["c"], np.array([3, 123, 123]))


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "test_ingest_csv_dense_array")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_csv_dense_array(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a dense array from a CSV file using ingest_csv().
    """
    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        sparse=False,
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                A.df[:][attribute],
                np.array([row * 10 + col for row in range(1, 21)]),
            )


@pytest.mark.skip
@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "test_ingest_csv_dense_array_apppend")],
    indirect=["udf_uri", "array_name"],
)
def test_ingest_csv_dense_array_apppend(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/increment.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        mode="ingest",
        full_domain=True,
        sparse=False,
        name=udf_uri,  # "unittest/test_ingest_csv" --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        assert A.nonempty_domain() == ((0, 19),)

    tiledb.cloud.udf.exec(
        f"s3://{bucket}/inputs/{array_name}.csv",
        f"tiledb://{namespace}/s3://{bucket}/{array_name}.tdb",
        key,
        secret,
        mode="append",
        row_start_idx=20,
        name=udf_uri,  # unittest/test_ingest_csv --> TileDB-Inc/ingest_csv
    )

    time.sleep(10)

    with tiledb.open(
        f"tiledb://{namespace}/{array_name}.tdb", ctx=tiledb.Ctx(config)
    ) as A:
        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                A.df[:][attribute],
                np.array([row * 10 + col for row in range(1, 21)] * 2),
            )
