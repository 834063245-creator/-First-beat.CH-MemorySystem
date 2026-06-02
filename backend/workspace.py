"""工作区工具已移除 — 开源版不需要。"""
import logging

logger = logging.getLogger(__name__)


def read_file(path: str) -> str:
    logger.debug("read_file 不可用（开源版）")
    return ""


def list_files(path: str = ".") -> list:
    logger.debug("list_files 不可用（开源版）")
    return []


def grep_files(pattern: str, path: str = ".") -> list:
    logger.debug("grep_files 不可用（开源版）")
    return []


def write_file(path: str, content: str) -> None:
    logger.debug("write_file 不可用（开源版）")


def edit_file(path: str, old: str, new: str) -> None:
    logger.debug("edit_file 不可用（开源版）")


def bash(command: str) -> str:
    logger.debug("bash 不可用（开源版）")
    return ""


def glob(pattern: str) -> list:
    logger.debug("glob 不可用（开源版）")
    return []
