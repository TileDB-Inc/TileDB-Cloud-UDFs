import tiledb
import tiledb.cloud

import public_udfs

import inspect
import types


def add_all_public_udfs():
    """
    Register or update all the UDFs available in the Public UDFs module.
    """
    for _, fnc in inspect.getmembers(public_udfs, inspect.isfunction):
        add_public_udf(fnc)


def add_public_udf(fnc, namespace="TileDB-Inc"):
    """
    Register or update the UDF for TileDB-Inc.
    """
    # Pass empty globals dictionary. Functions defined outside of __main__
    # are not supported by Cloudpickle, but TileDB public UDFs are defined in
    # public_udfs.py. This workaround allows the UDF to be unpickled by
    # striping the public_udf module.
    # https://stackoverflow.com/questions/49821323/python3-pickle-a-function-without-side-effects
    udf_code = types.FunctionType(fnc.__code__, {})
    udf_name = "{}".format(udf_code.__name__)

    if not tiledb.cloud.udf.info("{}/{}".format(namespace, udf_name)):
        tiledb.cloud.udf.register_generic_udf(udf_code, udf_name, namespace=namespace)
    else:
        tiledb.cloud.udf.update_generic_udf(udf_code, udf_name, namespace=namespace)
