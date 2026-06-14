"""测试 app/tools/workspace.py — 工作区文件操作工具。

覆盖：read_file / write_file / edit_file / list_files / grep_files。
"""
import os
import tempfile
from app.tools.workspace import read_file, write_file, edit_file, list_files, grep_files, _resolve_path


class TestResolvePath:
    def test_relative_becomes_absolute(self):
        resolved = _resolve_path("foo.txt")
        assert os.path.isabs(resolved)

    def test_absolute_passthrough(self):
        # 用当前平台的绝对路径，跨平台 CI 兼容
        abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "foo.txt"))
        resolved = _resolve_path(abs_path)
        assert resolved == abs_path


class TestReadFile:
    def test_reads_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello world")
            fpath = f.name
        try:
            result = read_file(fpath)
            assert result == "hello world"
        finally:
            os.unlink(fpath)

    def test_file_not_found(self):
        result = read_file("/nonexistent/path/xyz.txt")
        assert "不存在" in result

    def test_unicode_decode_fallback_gbk(self):
        """UTF-8 解码失败时自动回退 GBK。"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            # 写入 GBK 编码的中文
            f.write("中文测试".encode("gbk"))
            fpath = f.name
        try:
            result = read_file(fpath)
            assert "中文测试" in result
        finally:
            os.unlink(fpath)


class TestWriteFile:
    def test_writes_and_returns_size(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "test.txt")
            result = write_file(fpath, "hello")
            assert "已写入" in result
            with open(fpath, "r", encoding="utf-8") as f:
                assert f.read() == "hello"

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "sub", "deep", "file.txt")
            write_file(fpath, "nested")
            assert os.path.exists(fpath)


class TestEditFile:
    def test_replaces_unique_string(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "edit.txt")
            write_file(fpath, "line1\nline2\nline3")
            result = edit_file(fpath, "line2", "replaced")
            assert "已修改" in result
            content = read_file(fpath)
            assert "replaced" in content
            assert "line2" not in content

    def test_duplicate_match_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "edit.txt")
            write_file(fpath, "dup\nsomething\ndup")
            result = edit_file(fpath, "dup", "new")
            assert "出现了 2 次" in result

    def test_not_found_reports_error(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "edit.txt")
            write_file(fpath, "hello")
            result = edit_file(fpath, "nonexistent", "new")
            assert "未找到" in result


class TestListFiles:
    def test_matches_py_files(self):
        matches = list_files("*.py")
        assert isinstance(matches, str)
        assert len(matches) > 0

    def test_no_match(self):
        matches = list_files("*.xyznonexistent")
        assert "未匹配到" in matches


class TestGrepFiles:
    def test_finds_pattern_in_py_files(self):
        result = grep_files("def test_", "tests/*.py")
        assert isinstance(result, str)
        assert len(result) > 0
        # 应包含行号
        assert ":" in result

    def test_invalid_regex(self):
        result = grep_files("[invalid", "*.py")
        assert "正则表达式错误" in result

    def test_no_files_matched(self):
        result = grep_files("xxx", "*.nonexistentpattern")
        assert "未匹配到文件" in result

    def test_no_content_matched(self):
        result = grep_files("xyz_absolutely_no_match_99999", "tests/test_paths.py")
        assert "未找到匹配" in result
