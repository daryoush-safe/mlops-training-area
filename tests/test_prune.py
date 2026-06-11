import pytest

from sqlgen.data.prune import ExtractionError, extract_used_schema
from sqlgen.data.schema import Column, DBSchema, ForeignKey, Table


@pytest.fixture
def schema() -> DBSchema:
    return DBSchema(
        db_id="concert_singer",
        tables=[
            Table(
                name="singer",
                columns=[
                    Column(name="Singer_ID", type="number", is_primary_key=True),
                    Column(name="Name", type="text"),
                    Column(name="Country", type="text"),
                    Column(name="Age", type="number"),
                ],
            ),
            Table(
                name="concert",
                columns=[
                    Column(name="Concert_ID", type="number", is_primary_key=True),
                    Column(name="Theme", type="text"),
                    Column(name="Stadium_ID", type="number"),
                    Column(name="Year", type="text"),
                ],
            ),
            Table(
                name="singer_in_concert",
                columns=[
                    Column(name="Concert_ID", type="number", is_primary_key=True),
                    Column(name="Singer_ID", type="number"),
                ],
            ),
        ],
        foreign_keys=[
            ForeignKey(
                table="singer_in_concert",
                column="Singer_ID",
                ref_table="singer",
                ref_column="Singer_ID",
            ),
        ],
    )


def test_count_star_only_marks_table(schema):
    result = extract_used_schema("SELECT count(*) FROM singer", schema)
    assert result.method == "parser"
    assert result.tables == ["singer"]
    assert result.used_columns["singer"] == set()


def test_unqualified_columns_resolve(schema):
    result = extract_used_schema("SELECT name, country FROM singer WHERE age > 20", schema)
    assert result.used_columns == {"singer": {"name", "country", "age"}}


def test_join_with_spider_aliases(schema):
    sql = (
        "SELECT T2.name FROM singer_in_concert AS T1 "
        "JOIN singer AS T2 ON T1.singer_id = T2.singer_id "
        "WHERE T1.concert_id = 1"
    )
    result = extract_used_schema(sql, schema)
    assert result.used_columns == {
        "singer": {"name", "singer_id"},
        "singer_in_concert": {"singer_id", "concert_id"},
    }


def test_double_quoted_literal_not_treated_as_column(schema):
    result = extract_used_schema('SELECT name FROM singer WHERE country = "France"', schema)
    assert result.used_columns == {"singer": {"name", "country"}}


def test_select_star_expands_to_all_columns(schema):
    result = extract_used_schema("SELECT * FROM concert", schema)
    assert result.used_columns["concert"] == {"concert_id", "theme", "stadium_id", "year"}


def test_subquery_columns_attributed_to_inner_table(schema):
    sql = (
        "SELECT name FROM singer WHERE singer_id IN "
        "(SELECT singer_id FROM singer_in_concert)"
    )
    result = extract_used_schema(sql, schema)
    assert result.used_columns == {
        "singer": {"name", "singer_id"},
        "singer_in_concert": {"singer_id"},
    }


def test_set_operation(schema):
    sql = "SELECT name FROM singer UNION SELECT theme FROM concert"
    result = extract_used_schema(sql, schema)
    assert result.used_columns == {"singer": {"name"}, "concert": {"theme"}}


def test_unparseable_sql_uses_fallback(schema):
    result = extract_used_schema("SELEC name FRM singer WHERE", schema)
    assert result.method == "fallback"
    assert "singer" in result.used_columns


def test_unparseable_sql_raises_without_fallback(schema):
    with pytest.raises(ExtractionError):
        extract_used_schema("not sql at all", schema, enable_fallback=False)


def test_prune_keeps_order_pks_and_fks(schema):
    result = extract_used_schema(
        "SELECT T2.name FROM singer_in_concert AS T1 "
        "JOIN singer AS T2 ON T1.singer_id = T2.singer_id",
        schema,
    )
    pruned = schema.prune(result.used_columns)
    assert [t.name for t in pruned.tables] == ["singer", "singer_in_concert"]
    singer = pruned.tables[0]
    # PK retained even though not in SELECT, column order follows the full schema
    assert singer.column_names() == ["Singer_ID", "Name"]
    assert len(pruned.foreign_keys) == 1


def test_prune_star_keeps_all_columns(schema):
    result = extract_used_schema("SELECT * FROM concert", schema)
    pruned = schema.prune(result.used_columns)
    assert pruned.tables[0].column_names() == ["Concert_ID", "Theme", "Stadium_ID", "Year"]
