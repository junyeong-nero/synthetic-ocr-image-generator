# -*- coding: utf-8 -*-
import sys
from collections import deque
from typing import Dict, List, Any
from lxml import etree
from apted import APTED, Config


class TableTree(object):
    """Represents a node in a table tree structure."""

    def __init__(self, tag, colspan=None, rowspan=None, content=None):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list()

    def __repr__(self):
        return f"Node({self.tag})"


class CustomConfig(Config):
    """Custom configuration for the APTED algorithm to compare table tree nodes."""

    def rename(self, node1, node2):
        """Compares attributes of two nodes."""
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0
        return 0.0

    def children(self, node):
        """Get children of a node."""
        return node.children


class TEDS(object):
    """
    Tree-Edit Distance-based Similarity (TEDS) for tables.
    """

    def __init__(self, structure_only=False):
        self.structure_only = structure_only
        self.parser = etree.HTMLParser(recover=True)

    def _html_to_tree(self, html_string: str) -> TableTree:
        """Converts an HTML string to a TableTree."""
        root = etree.fromstring(html_string, self.parser)
        # Find the first `<table>` element
        table_element = root.find(".//table")
        if table_element is None:
            return TableTree("table")  # Return an empty table if no table tag is found

        # Recursively build the tree
        return self._build_tree_recursive(table_element)

    def _build_tree_recursive(self, element) -> TableTree:
        """Recursively builds a TableTree from an lxml element."""
        colspan = int(element.get("colspan", "1"))
        rowspan = int(element.get("rowspan", "1"))
        content = "" if self.structure_only else (element.text or "").strip()

        node = TableTree(
            tag=element.tag, colspan=colspan, rowspan=rowspan, content=content
        )

        for child_element in element:
            node.children.append(self._build_tree_recursive(child_element))

        return node

    def evaluate(self, pred_html: str, true_html: str) -> Dict[str, Any]:
        """
        Calculates the TEDS score between a predicted and a true HTML table.
        """
        try:
            true_tree = self._html_to_tree(true_html)
            pred_tree = self._html_to_tree(pred_html)
        except Exception as e:
            return {"teds": 0.0, "error": str(e)}

        # Calculate the edit distance
        apted = APTED(true_tree, pred_tree, CustomConfig())
        edit_distance = apted.compute_edit_distance()

        # Normalize to get similarity score
        size1 = self._get_tree_size(true_tree)
        size2 = self._get_tree_size(pred_tree)

        if size1 + size2 == 0:
            teds_score = 1.0 if edit_distance == 0 else 0.0
        else:
            teds_score = 1.0 - (edit_distance / (size1 + size2))

        return {"teds": teds_score}

    def _get_tree_size(self, tree: TableTree) -> int:
        """Calculates the number of nodes in a tree."""
        count = 1
        for child in tree.children:
            count += self._get_tree_size(child)
        return count


if __name__ == "__main__":
    # Example Usage
    true_table_html = """
    <html>
    <body>
        <table>
            <thead>
                <tr><th>Header 1</th><th>Header 2</th></tr>
            </thead>
            <tbody>
                <tr><td>Data 1</td><td>Data 2</td></tr>
                <tr><td>Data 3</td><td>Data 4</td></tr>
            </tbody>
        </table>
    </body>
    </html>
    """

    pred_table_html_perfect = """
    <table>
        <thead>
            <tr><th>Header 1</th><th>Header 2</th></tr>
        </thead>
        <tbody>
            <tr><td>Data 1</td><td>Data 2</td></tr>
            <tr><td>Data 3</td><td>Data 4</td></tr>
        </tbody>
    </table>
    """

    pred_table_html_imperfect = """
    <table>
        <tbody>
            <tr><td>Data 1</td><td>Data 2</td></tr>
            <tr><td>Data 3</td><td colspan="2">Data 4</td></tr>
        </tbody>
    </table>
    """

    pred_table_html_empty = "<table></table>"

    teds_calculator = TEDS(structure_only=True)

    # Perfect match
    score_perfect = teds_calculator.evaluate(pred_table_html_perfect, true_table_html)
    print(f"TEDS score (perfect match): {score_perfect['teds']:.4f}")

    # Imperfect match
    score_imperfect = teds_calculator.evaluate(
        pred_table_html_imperfect, true_table_html
    )
    print(f"TEDS score (imperfect match): {score_imperfect['teds']:.4f}")

    # Empty prediction
    score_empty = teds_calculator.evaluate(pred_table_html_empty, true_table_html)
    print(f"TEDS score (empty prediction): {score_empty['teds']:.4f}")
