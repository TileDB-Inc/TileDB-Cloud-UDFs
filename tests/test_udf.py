import tiledb
import tiledb.cloud
import public_udfs

from contextlib import suppress
import numpy as np
from numpy.testing import assert_array_equal
import os
import pandas
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

    if not tiledb.cloud.udf.list_registered_udfs(
        namespace, test_udf_name
    ).udf_info_list:
        tiledb.cloud.udf.register_generic_udf(udf, test_udf_name)
    else:
        tiledb.cloud.udf.update_generic_udf(udf, test_udf_name)

    return "{}/{}".format(namespace, test_udf_name)

    # TODO should delete the test UDF at teardown. But this is erroring out for
    # me at the moment.
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
    tiledb_uri = "tiledb://{}/{}.tdb".format(namespace, array_name)
    s3_uri = "s3://{}/{}.tdb".format(bucket, array_name)

    # Supressing errors is a temporary solution to delays between writing (or in
    # this case, deleting) and reading arrays on S3.
    with suppress(Exception):
        tiledb.cloud.deregister_array(tiledb_uri)

    with suppress(Exception):
        tiledb.remove(s3_uri)

    yield

    with suppress(Exception):
        tiledb.cloud.deregister_array(tiledb_uri)

    with suppress(Exception):
        tiledb.remove(s3_uri)


@pytest.mark.parametrize(
    "udf_uri,array_name", [(public_udfs.ingest_csv, "increment")], indirect=["udf_uri"],
)
def test_ingest_csv_sparse_array(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a sparse array from a CSV file using ingest_csv().
    """
    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        name=udf_uri,  # TileDB-Inc/test-ingest_csv
    )

    time.sleep(10)

    with tiledb.SparseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])

        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                data[attribute], np.array([row * 10 + col for row in range(1, 21)]),
            )


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "increment_sparse")],
    indirect=["udf_uri"],
)
def test_ingest_csv_sparse_array_apppend(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a sparse array from a CSV file using ingest_csv() in the default
    ingest mode and then append additional data to it using the append mode.
    """
    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment_sparse1"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        mode="ingest",
        full_domain=True,
        index_col=("x"),
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.SparseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])
        number_of_rows = data.shape[0]
        assert number_of_rows == 20

    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment_sparse2"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        mode="append",
        full_domain=True,
        index_col=("x"),
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.SparseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])

        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                data[attribute], np.array([row * 10 + col for row in range(1, 21)] * 2),
            )


@pytest.mark.parametrize(
    "udf_uri,array_name",
    [(public_udfs.ingest_csv, "increment_sparse")],
    indirect=["udf_uri"],
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
        "s3://{}/inputs/{}.csv".format(bucket, "increment_sparse1"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        mode="ingest",
        full_domain=True,
        index_col=("x"),
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.SparseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])
        number_of_rows = data.shape[0]
        assert number_of_rows == 20

    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment_sparse2_mismatch"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        mode="append",
        full_domain=True,
        index_col=("x"),
        header=0,
        names=["x", "c", "b", "a"],
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.SparseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])

        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                data[attribute], np.array([row * 10 + col for row in range(1, 21)] * 2),
            )


@pytest.mark.parametrize(
    "udf_uri,array_name", [(public_udfs.ingest_csv, "increment")], indirect=["udf_uri"],
)
def test_ingest_csv_sparse_array_null_replace(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    From a CSV file containing NaNs, produce a sparse array using ingest_csv()
    where the NaNs are replaced with the value given by fillna.
    """
    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment_nulls"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        fillna=123,
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.SparseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])

        assert_array_equal(data["a"], np.array([1, 1, 1]))
        assert_array_equal(data["b"], np.array([2, 2, 123]))
        assert_array_equal(data["c"], np.array([3, 123, 123]))


@pytest.mark.parametrize(
    "udf_uri,array_name", [(public_udfs.ingest_csv, "increment")], indirect=["udf_uri"],
)
def test_ingest_csv_dense_array(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    """
    Create a dense array from a CSV file using ingest_csv().
    """
    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        sparse=False,
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.DenseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])

        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                data[attribute], np.array([row * 10 + col for row in range(1, 21)]),
            )


@pytest.mark.skip
@pytest.mark.parametrize(
    "udf_uri,array_name", [(public_udfs.ingest_csv, "increment")], indirect=["udf_uri"],
)
def test_ingest_csv_dense_array_apppend(
    udf_uri, array_name, key, secret, namespace, bucket, config
):
    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, "increment"),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        mode="ingest",
        full_domain=True,
        sparse=False,
        name=udf_uri,  # "TileDB-Inc/test-ingest_csv"
    )

    time.sleep(10)

    with tiledb.DenseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])
        number_of_rows = data.shape[0]
        assert number_of_rows == 20

    tiledb.cloud.udf.exec(
        "s3://{}/inputs/{}.csv".format(bucket, array_name),
        "s3://{}/{}.tdb".format(bucket, array_name),
        namespace,
        key,
        secret,
        mode="append",
        row_start_idx=number_of_rows,
        name=udf_uri,  # TileDB-Inc/test-ingest_csv
    )

    time.sleep(10)

    with tiledb.DenseArray(
        "tiledb://{}/{}.tdb".format(namespace, array_name), "r", ctx=tiledb.Ctx(config)
    ) as A:
        data = pandas.DataFrame(A[:])

        for col, attribute in enumerate(("a", "b", "c"), 1):
            assert_array_equal(
                data[attribute], np.array([row * 10 + col for row in range(1, 21)] * 2),
            )

