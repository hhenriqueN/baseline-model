import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controller.protocol import build_protocol_xml, encode_command_line, decode_telemetry_line, \
    OUTPUT_FIELDS, INPUT_FIELDS


def test_protocol_xml_contains_every_field_node():
    xml = build_protocol_xml()
    for c in OUTPUT_FIELDS + INPUT_FIELDS:
        assert f"<node>{c.node}</node>" in xml


def test_encode_command_line_round_trips_order():
    values = {c.name: i * 0.1 for i, c in enumerate(INPUT_FIELDS)}
    line = encode_command_line(values).decode("ascii")
    parts = line.strip().split(",")
    assert len(parts) == len(INPUT_FIELDS)
    for i, p in enumerate(parts):
        assert abs(float(p) - i * 0.1) < 1e-9


def test_encode_command_line_missing_field_defaults_zero():
    line = encode_command_line({}).decode("ascii")
    parts = line.strip().split(",")
    assert all(float(p) == 0.0 for p in parts)


def test_decode_telemetry_line_round_trip():
    fake_values = []
    for c in OUTPUT_FIELDS:
        fake_values.append("1" if c.type == "int" else "3.14159")
    line = ",".join(fake_values)
    decoded = decode_telemetry_line(line)
    assert decoded is not None
    for c in OUTPUT_FIELDS:
        if c.type == "int":
            assert decoded[c.name] == 1
        else:
            assert abs(decoded[c.name] - 3.14159) < 1e-6


def test_decode_telemetry_line_wrong_field_count_returns_none():
    assert decode_telemetry_line("1,2,3") is None


def test_decode_telemetry_line_garbage_returns_none():
    garbage = ",".join(["not_a_number"] * len(OUTPUT_FIELDS))
    assert decode_telemetry_line(garbage) is None


def test_protocol_xml_is_well_formed_xml():
    # FlightGear refuses the whole channel on any XML error (e.g. "--" inside a comment)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(build_protocol_xml())
    assert root.find("generic/output") is not None and root.find("generic/input") is not None
    assert len(root.findall("generic/output/chunk")) == len(OUTPUT_FIELDS)
    assert len(root.findall("generic/input/chunk")) == len(INPUT_FIELDS)


def test_every_chunk_has_explicit_format():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(build_protocol_xml())
    for ch in root.iter("chunk"):
        fmt = ch.findtext("format")
        assert fmt, f"chunk {ch.findtext('name')} has no <format>"
        assert (fmt == "%d") == (ch.findtext("type") == "int")
