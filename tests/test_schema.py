from sqlgen.data.schema import schema_from_spider_entry, serialize_schema

SPIDER_ENTRY = {
    "db_id": "perpetrator",
    "table_names_original": ["perpetrator", "people"],
    "table_names": ["perpetrator", "people"],
    "column_names_original": [
        [-1, "*"],
        [0, "Perpetrator_ID"],
        [0, "People_ID"],
        [1, "People_ID"],
        [1, "Name"],
    ],
    "column_names": [
        [-1, "*"],
        [0, "perpetrator id"],
        [0, "people id"],
        [1, "people id"],
        [1, "name"],
    ],
    "column_types": ["text", "number", "number", "number", "text"],
    "primary_keys": [1, 3],
    "foreign_keys": [[2, 3]],
}


def test_schema_from_spider_entry():
    schema = schema_from_spider_entry(SPIDER_ENTRY)
    assert schema.db_id == "perpetrator"
    assert [t.name for t in schema.tables] == ["perpetrator", "people"]
    perp = schema.tables[0]
    assert perp.column_names() == ["Perpetrator_ID", "People_ID"]
    assert perp.columns[0].is_primary_key
    assert not perp.columns[1].is_primary_key
    fk = schema.foreign_keys[0]
    assert (fk.table, fk.column, fk.ref_table, fk.ref_column) == (
        "perpetrator", "People_ID", "people", "People_ID",
    )


def test_serialize_compact():
    schema = schema_from_spider_entry(SPIDER_ENTRY)
    text = serialize_schema(schema, style="compact")
    assert text == "perpetrator(Perpetrator_ID, People_ID) | people(People_ID, Name)"


def test_serialize_verbose():
    schema = schema_from_spider_entry(SPIDER_ENTRY)
    text = serialize_schema(schema, style="verbose")
    assert "table perpetrator (Perpetrator_ID number primary key, People_ID number)" in text
    assert "foreign key perpetrator.People_ID references people.People_ID" in text


def test_sqlglot_mapping_types():
    schema = schema_from_spider_entry(SPIDER_ENTRY)
    mapping = schema.to_sqlglot_mapping()
    assert mapping["people"]["Name"] == "TEXT"
    assert mapping["perpetrator"]["Perpetrator_ID"] == "NUMERIC"
