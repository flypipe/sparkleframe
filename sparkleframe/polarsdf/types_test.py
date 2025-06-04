import pytest
import pyspark.sql.types as pst
from sparkleframe.polarsdf import types as sft


class TestTypes:
    @pytest.mark.parametrize(
        "spark_type, sf_type",
        [
            (pst.StringType(), sft.StringType()),
            (pst.IntegerType(), sft.IntegerType()),
            (pst.LongType(), sft.LongType()),
            (pst.FloatType(), sft.FloatType()),
            (pst.DoubleType(), sft.DoubleType()),
            (pst.BooleanType(), sft.BooleanType()),
            (pst.DateType(), sft.DateType()),
            (pst.TimestampType(), sft.TimestampType()),
            (pst.ByteType(), sft.ByteType()),
            (pst.ShortType(), sft.ShortType()),
            (pst.BinaryType(), sft.BinaryType()),
        ],
    )
    def test_simple_type_equivalence(self, spark_type, sf_type):
        assert spark_type.typeName() == sf_type.typeName()
        assert spark_type.simpleString() == sf_type.simpleString()
        assert spark_type.jsonValue() == sf_type.jsonValue()

    @pytest.mark.parametrize("precision, scale", [(10, 2), (5, 0), (20, 10)])
    def test_decimal_type_equivalence(self, precision, scale):
        spark_type = pst.DecimalType(precision, scale)
        sf_type = sft.DecimalType(precision, scale)

        assert spark_type.typeName() == sf_type.typeName()
        assert spark_type.simpleString() == sf_type.simpleString()
        assert spark_type.jsonValue() == sf_type.jsonValue()
        assert sf_type.precision == spark_type.precision
        assert sf_type.scale == spark_type.scale

    def test_struct_type_equivalence(self):
        sf_struct = sft.StructType(
            [
                sft.StructField("id", sft.IntegerType(), True),
                sft.StructField("name", sft.StringType(), False),
            ]
        )

        ps_struct = pst.StructType(
            [
                pst.StructField("id", pst.IntegerType(), True),
                pst.StructField("name", pst.StringType(), False),
            ]
        )

        assert sf_struct.typeName() == ps_struct.typeName()
        assert isinstance(sf_struct.fields[0].dataType, sft.IntegerType)
        assert sf_struct.fields[0].name == ps_struct.fields[0].name
        assert sf_struct.fields[1].nullable == ps_struct.fields[1].nullable

    def test_struct_field_methods(self):
        sf = sft.StructField("name", sft.StringType(), False, {"meta": 1})
        psf = pst.StructField("name", pst.StringType(), False, {"meta": 1})

        assert sf.name == psf.name
        assert sf.nullable == psf.nullable
        assert sf.dataType.typeName() == psf.dataType.typeName()
        assert sf.simpleString() == psf.simpleString()
        assert sf.__repr__() == psf.__repr__()
        assert sf.jsonValue() == psf.jsonValue()

    def test_struct_type_methods(self):
        sf1 = sft.StructField("id", sft.IntegerType(), True)
        sf2 = sft.StructField("name", sft.StringType(), False)
        sftype = sft.StructType([sf1, sf2])

        psf1 = pst.StructField("id", pst.IntegerType(), True)
        psf2 = pst.StructField("name", pst.StringType(), False)
        pstype = pst.StructType([psf1, psf2])

        # simpleString
        assert sftype.simpleString() == pstype.simpleString()

        # repr
        assert repr(sftype) == repr(pstype)

        # jsonValue
        assert sftype.jsonValue() == pstype.jsonValue()

        # __len__
        assert len(sftype) == 2

        # __getitem__ by index
        assert isinstance(sftype[0], sft.StructField)
        assert sftype[0].name == pstype[0].name

        # __getitem__ by name
        assert sftype["name"].dataType.typeName() == pstype["name"].dataType.typeName()

        # __getitem__ slice
        sliced = sftype[0:1]
        assert isinstance(sliced, sft.StructType)
        assert len(sliced) == 1
        assert sliced[0].name == "id"

        sliced = pstype[0:1]
        assert isinstance(sliced, pst.StructType)
        assert len(sliced) == 1
        assert sliced[0].name == "id"

        # __iter__
        assert [f.name for f in sftype] == [f.name for f in pstype]

        # fieldNames
        assert sftype.fieldNames() == pstype.fieldNames()

    def test_struct_type_getitem_errors(self):
        sftype = sft.StructType([sft.StructField("a", sft.StringType())])

        with pytest.raises(KeyError):
            _ = sftype["missing"]

        with pytest.raises(IndexError):
            _ = sftype[99]

        with pytest.raises(ValueError):
            _ = sftype[{"bad": "key"}]
