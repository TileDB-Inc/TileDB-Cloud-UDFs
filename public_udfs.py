def ingest_csv(source_csv, target, key, secret, **kwargs):
    """
    Create TileDB array with an input CSV file located in a given S3 Bucket at source_csv. The S3
    Bucket key and secret credentials must be provided. Outputs a target_array in the given S3
    Bucket and namespace on TileDB Cloud.

    :param source_csv: S3 Bucket URI for the input CSV file.
    :param target: The URI of the output TileDB Cloud array in the style of
                   "tiledb://namespace/s3://bucket/array".
    :param key: The AWS Access Key ID for accessing the S3 Bucket.
    :param secret: The AWS Secret Access Key for accessing the S3 Bucket.
    :Keyword Arguments:
        - Any ``pandas.read_csv`` supported keyword argument.
        - TileDB-specific arguments:
            * ``allows_duplicates``: Generated schema should allow duplicates
            * ``cell_order``: Schema cell order
            * ``tile_order``: Schema tile order
            * ``mode``: (default ``ingest``), Ingestion mode:
                - ``ingest``: Create a new array
                - ``schema_only``
                - ``append``: Append data to an existing array. Sparse arrays require an index
                  column. Dense arrays require the ``full_domain`` mode and ``row_start_idx``.
            * ``full_domain``: Dimensions should be created with full range of the dtype
            * ``attrs_filters``: FilterList to apply to all Attributes
            * ``coords_filters``: FilterList to apply to all coordinates (Dimensions)
            * ``sparse``: (default True) Create sparse schema
            * ``tile``: Schema tiling (capacity)
            * ``fillna``: Replace NaN/NAs with a given value
            * ``date_spec``: Dictionary of {``column_name``: format_spec} to apply to date/time
              columns which are not correctly inferred by pandas 'parse_dates'.
              Format must be specified using the Python format codes:
              https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    :return: None

    **Example:**
    >>> import tiledb.cloud
    >>> # Create a sparse array from a CSV file
    >>> tiledb.cloud.udf.exec(
    ...     "s3://bucket/data1.csv",
    ...     "tiledb://namespace/s3://array.tdb",
    ...     aws_key,
    ...     aws_secret,
    ...     mode="ingest",
    ...     full_domain=True,
    ...     row_start_idx=("row"),
    ...     name="TileDB-Inc/ingest_csv",
    ... )
    >>> # Append additional data to the sparse array
    >>> tiledb.cloud.udf.exec(
    ...     "s3://bucket/data2.csv",
    ...     "tiledb://namespace/s3://array.tdb",
    ...     aws_key,
    ...     aws_secret,
    ...     mode="append",
    ...     row_start_idx=("row"),
    ...     name="TileDB-Inc/ingest_csv",
    ... )
    """

    # import tiledb and set up configured context with access to S3
    import tiledb

    config = {
        "vfs.s3.aws_access_key_id": key,
        "vfs.s3.aws_secret_access_key": secret,
    }

    tiledb.default_ctx(config)

    # ingest CSV, which will create array
    # - stored on S3
    # - visible in TileDB Console
    tiledb.from_csv(target, source_csv, **kwargs)

    return "done"
