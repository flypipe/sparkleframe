import re
from unittest import mock

import pytest

from scripts.calculate_version import get_changes, RE_COMMIT_TYPE, get_release_branches, calculate_version, \
    get_commit_list, get_commit_message
from scripts.generate_changelog import generate_changelog, save_changelog


class TestGenerateChangeLog:

    @pytest.mark.parametrize(
        "message,expected_message",
        [
            ("#3-set-up-publish-to-testpypi fix linter", {'3': '<a href="https://github.com/flypipe/sparkleframe/issues/3" '
      'target="_blank" rel="noopener noreferrer">3 Set up publish to '
      'testpypi</a>'}),
        ]
    )
    def test_calculate_version(self, message, expected_message):
        with (
            mock.patch('scripts.generate_changelog.get_release_branches', return_value=["origin/release/0.0.0"]),
            mock.patch('scripts.generate_changelog.get_commit_list', return_value=["1"]),
            mock.patch('scripts.generate_changelog.get_commit_message', return_value=message)
        ):
            issues = generate_changelog()
            assert issues == expected_message



