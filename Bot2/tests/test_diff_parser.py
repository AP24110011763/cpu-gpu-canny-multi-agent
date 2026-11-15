"""Unit tests for Unified Diff Parser."""

import unittest
from bot2.diff_parser import DiffParser, FileDiff, ChangedLine


SAMPLE_DIFF = """diff --git a/app.py b/app.py
index 1234567..89abcdef 100644
--- a/app.py
+++ b/app.py
@@ -10,4 +10,6 @@ def calculate():
     base = 100
-    return base * 2
+    bonus = 50
+    return (base * 2) + bonus
"""

SAMPLE_MULTI_DIFF = """diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def hello():
+    return "world"
diff --git a/deleted.py b/deleted.py
deleted file mode 100644
index 2222222..0000000
--- a/deleted.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def gone():
-    pass
"""


class TestDiffParser(unittest.TestCase):
    def test_parse_patch_hunks(self):
        patch = (
            "@@ -5,3 +5,4 @@\n"
            " context\n"
            "+added line 1\n"
            "+added line 2\n"
            " unchanged\n"
        )
        changed = DiffParser.parse_patch_hunks("foo.py", patch)
        self.assertEqual(len(changed), 2)
        self.assertEqual(changed[0].line_number, 6)
        self.assertEqual(changed[0].content, "added line 1")
        self.assertEqual(changed[1].line_number, 7)
        self.assertEqual(changed[1].content, "added line 2")

    def test_parse_full_unified_diff(self):
        file_diffs = DiffParser.parse_full_unified_diff(SAMPLE_DIFF)
        self.assertEqual(len(file_diffs), 1)
        fd = file_diffs[0]
        self.assertEqual(fd.filename, "app.py")
        self.assertEqual(fd.status, "modified")
        self.assertEqual(len(fd.changed_lines), 2)
        self.assertEqual(fd.changed_lines[0].line_number, 11)
        self.assertEqual(fd.changed_lines[0].content, "    bonus = 50")
        self.assertEqual(fd.changed_lines[1].line_number, 12)

    def test_parse_multi_file_diff(self):
        file_diffs = DiffParser.parse_full_unified_diff(SAMPLE_MULTI_DIFF)
        self.assertEqual(len(file_diffs), 2)

        new_f = file_diffs[0]
        self.assertEqual(new_f.filename, "new_file.py")
        self.assertEqual(new_f.status, "added")
        self.assertEqual(len(new_f.changed_lines), 2)

        del_f = file_diffs[1]
        self.assertEqual(del_f.filename, "deleted.py")
        self.assertEqual(del_f.status, "deleted")
        self.assertTrue(del_f.is_deleted)
        self.assertEqual(len(del_f.changed_lines), 0)

    def test_empty_diff(self):
        self.assertEqual(DiffParser.parse_full_unified_diff(""), [])
        self.assertEqual(DiffParser.parse_patch_hunks("f.py", ""), [])


if __name__ == "__main__":
    unittest.main()
