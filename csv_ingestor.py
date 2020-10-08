import tiledb, tiledb.cloud, numpy as np

# define ingestion function
def ingest_csv(args_list):
    # the UDF entrypoint currently only supports passing a single
    # argument when using JSON, so we need to unpack arguments here
    source_csv, target_array, namespace, key, secret = args_list[0]

    # import tiledb and set up configured context with access to S3
    import tiledb
    config = {'vfs.s3.aws_access_key_id': key,
              'vfs.s3.aws_secret_access_key': secret}
    
    tiledb.default_ctx(config)

    # ingest CSV, which will create array
    # - stored on S3
    # - visible in TileDB Console
    target_tiledb_array = f"tiledb://{namespace}/{target_array}"
    tiledb.from_csv(target_tiledb_array, source_csv)
    
    # `from_csv` also supports appending to an existing array
    # which will create a new TileDB fragment

    return 'done'