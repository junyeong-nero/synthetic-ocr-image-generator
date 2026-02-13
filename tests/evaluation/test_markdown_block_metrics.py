from metrics.markdown_block_metrics import evaluate_markdown_blocks, split_markdown_blocks


def test_split_markdown_blocks_preserves_text_table_formula_order() -> None:
    markdown = "Intro\n\n|A|B|\n|---|---|\n|1|2|\n\n$$x^2 + y^2$$\n\nTail"

    blocks = split_markdown_blocks(markdown)

    assert [block["type"] for block in blocks] == ["text", "table", "formula", "text"]


def test_evaluate_markdown_blocks_returns_per_type_scores() -> None:
    markdown = "Intro\n\n|A|B|\n|---|---|\n|1|2|\n\n$E=mc^2$"

    metrics = evaluate_markdown_blocks(markdown, markdown)

    assert metrics["markdown_text_score"] == 1.0
    assert metrics["markdown_table_teds"] == 1.0
    assert metrics["markdown_formula_score"] == 1.0
    assert metrics["markdown_order_score"] == 1.0
    assert metrics["markdown_overall_score"] == 1.0


def test_evaluate_markdown_blocks_order_score_drops_on_reordered_blocks() -> None:
    ground_truth = "alpha\n\n|A|B|\n|---|---|\n|1|2|\n\n$z$"
    prediction = "alpha\n\n$z$\n\n|A|B|\n|---|---|\n|1|2|"

    metrics = evaluate_markdown_blocks(prediction, ground_truth)

    assert metrics["markdown_order_score"] < 1.0
    assert metrics["markdown_overall_score"] < 1.0
