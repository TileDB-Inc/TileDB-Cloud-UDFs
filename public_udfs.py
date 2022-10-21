from typing import Dict, Optional, Sequence, Tuple, Union


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


def ingest_vcf_samples(
    array_uri: str,
    contig: Optional[str],
    sample_uris: Union[str, Sequence[str]],
    partition_idx_count: Tuple[int, int],
    tiledb_config: Optional[Dict[str, str]],
    memory_budget_mb: int,
    threads: int,
    stats: bool = False,
    resume: bool = False,
) -> Sequence[str]:
    """Returns a list of the URIs that were ingested by this job."""
    import tiledbvcf

    tiledbvcf.config_logging("info")
    print(f"tiledbvcf v{tiledbvcf.version}")

    print(f"Ingesting into array '{array_uri}'")
    if isinstance(sample_uris, str):
        sample_uris = [sample_uris]

    # open the array
    cfg = tiledbvcf.ReadConfig(tiledb_config=tiledb_config)
    ds = tiledbvcf.Dataset(array_uri, mode="w", cfg=cfg, stats=stats)
    print(f"Opened {array_uri} (schema v{ds.schema_version()})")
    # sample partition index/number
    sp_i, sp_n = partition_idx_count
    this_shard = sample_uris[sp_i::sp_n]
    if sp_n != 1:
        print(f"Processing sample partition {sp_i} of {sp_n}")
        print(f"...this partition includes {len(this_shard)} samples")
    ds.ingest_samples(
        sample_uris=this_shard,
        total_memory_budget_mb=memory_budget_mb,
        threads=threads,
        contig_mode="separate" if contig else "merged",
        **{"contigs_to_keep_separate": [contig]} if contig else {},
        resume=resume,
    )
    return this_shard
