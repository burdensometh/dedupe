.. _field_adapters:

Input Field Adapters
====================

Two data sources rarely name their fields the same way. A record from
one source might look like::

    {"zip": 2138, "street": "main st", "name": "acme"}

while the matching record from another source looks like::

    {"zipcode": "02138", "address": "MAIN ST", "name": "ACME"}

The field names differ (``zip``/``zipcode``, ``street``/``address``),
the zip code has lost its leading zero on one side, and the casing is
different. A :class:`dedupe.SourceAdapter` declares, once, how each
source maps onto the *common* field names of the variable definitions.

The adapter is part of the training configuration:

* it is validated against the data model when the matcher is built,
* it is saved inside the settings file by ``write_settings`` and
  reloaded unchanged by the ``Static`` matcher, and
* the exact same null, case and field-type semantics are applied at
  both training time and matching time.

No conversion rules are inferred from the data while matching. If a
source field has been renamed or a value has an incompatible type, an
:class:`dedupe.AdaptationError` is raised with the field and value that
could not be converted.

Declaring the adapters
----------------------

.. code-block:: python

    import dedupe

    variables = [
        dedupe.variables.String("Zip", has_missing=True),
        dedupe.variables.String("Street", has_missing=True),
        dedupe.variables.String("Name"),
    ]

    source_a = dedupe.SourceAdapter(
        name="source_a",
        fields=(
            # zip codes arrive as integers, leading zeros already lost;
            # pad declares the fixed-width string representation both
            # sides will use for blocking and distance calculation.
            dedupe.FieldSpec("zip", "Zip", type="str", pad=5),
            dedupe.FieldSpec("street", "Street", type="str", case="upper"),
            dedupe.FieldSpec("name", "Name", type="str", case="upper"),
        ),
    )

    source_b = dedupe.SourceAdapter(
        name="source_b",
        fields=(
            dedupe.FieldSpec("zipcode", "Zip", type="str", pad=5),
            dedupe.FieldSpec("address", "Street", type="str", case="upper"),
            dedupe.FieldSpec("name", "Name", type="str", case="upper"),
        ),
    )

    linker = dedupe.RecordLink(
        variables, adapters=(source_a, source_b)
    )

A single-source :class:`dedupe.Dedupe` takes one adapter, a
:class:`dedupe.RecordLink` or :class:`dedupe.Gazetteer` takes two, one
per data source. From that point on every method
(``prepare_training``, ``mark_pairs``, ``pairs``, ``join``,
``partition``, ``index``, ``search`` ...) accepts the raw source
records; adaptation happens internally before blocking and distance
calculation.

Field semantics
---------------

Each :class:`dedupe.FieldSpec` pins down one semantics for one common
field, and every source must use the same one:

``type``
    ``"str"``, ``"int"`` or ``"float"``. A value that does not match the
    declared type raises :class:`dedupe.AdaptationError`; numeric
    strings are never silently parsed into numbers. An integer fed into
    a declared string field is stringified (this is how a padded zip
    field recovers its leading zeros); booleans and floats are rejected.

``case``
    ``"upper"``, ``"lower"`` or ``"none"`` (the default). Applied only
    to string fields, identically on both sides.

``pad``
    Zero-pad digit strings to this width, e.g. ``pad=5`` normalizes
    both ``2138`` and ``"02138"`` to ``"02138"``.

``null_values``
    A ``frozenset`` of values treated as missing, defaulting to
    ``frozenset({None, ""})``. Every configured null becomes the single
    null marker ``None``, which is the only null semantics the blocking
    predicates and comparators see.

Recovering the original values
------------------------------

Adapted records contain the common fields and a verbatim copy of the
raw record under :data:`dedupe.ORIGINAL_RECORD_KEY`. Record ids are
never changed, so a link result can always be traced back to the
original input:

.. code-block:: python

    static_linker = dedupe.StaticRecordLink(settings_file)
    for (id_a, record_a), (id_b, record_b) in static_linker.pairs(data_1, data_2):
        raw_a = dedupe.original_record(record_a)
        raw_b = dedupe.original_record(record_b)

:class:`dedupe.SourceAdapter` objects and :class:`dedupe.FieldSpec`
objects are plain, immutable and picklable, so they round-trip through
the settings file without extra configuration.
