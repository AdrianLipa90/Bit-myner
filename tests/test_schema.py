import json
from pathlib import Path
import jsonschema

from bitmyner.marker_graph import HashEvidenceGraph


def test_graph_schema_accepts_empty_graph_shape():
    schema = json.loads(Path('schemas/MARKER_GRAPH_V0_1.schema.json').read_text())
    graph = HashEvidenceGraph().to_dict()
    jsonschema.validate(graph, schema)
