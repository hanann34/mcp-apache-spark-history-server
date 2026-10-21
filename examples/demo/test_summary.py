"""Facts extracted from Spark MCP payloads stay readable and faithful."""

from examples.demo.summary import parse_tool_text, summarize_tool


def test_parse_concatenated_application_json() -> None:
    raw = (
        '{"id": "spark-a", "name": "One", "attempts": '
        '[{"duration": 1500, "completed": true, "appSparkVersion": "3.5.3"}]}\n'
        '{"id": "spark-b", "name": "Two", "attempts": '
        '[{"duration": 2500, "completed": true, "appSparkVersion": "3.5.3"}]}'
    )
    wrapped = json_string(raw)
    apps = parse_tool_text(wrapped)
    assert [app["id"] for app in apps] == ["spark-a", "spark-b"]


def test_summarize_applications_uses_attempt_duration() -> None:
    text = summarize_tool(
        "list_applications",
        [
            {
                "id": "spark-a",
                "name": "Taxi",
                "attempts": [
                    {
                        "duration": 508614,
                        "completed": True,
                        "appSparkVersion": "3.5.3",
                    }
                ],
            }
        ],
    )
    assert "Taxi" in text
    assert "508.6 s" in text
    assert "3.5.3" in text


def test_environment_summary_ignores_identity_fields() -> None:
    text = summarize_tool(
        "compare_job_environments",
        {
            "spark_properties": {
                "common": {"spark.driver.memory": {}},
                "different": {
                    "spark.app.id": {"app1": "a", "app2": "b"},
                    "spark.driver.host": {"app1": "h1", "app2": "h2"},
                },
            }
        },
    )
    assert "1 Spark settings match" in text
    assert "host" in text
    assert "spark.driver.host" not in text


def json_string(value: str) -> str:
    import json

    return json.dumps(value)
