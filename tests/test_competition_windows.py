"""Portable resource cleanup required by the documented Windows MCP setup."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from competition_agent.service import Lab, source_hashes


class WindowsRuntimeTests(unittest.TestCase):
    def test_connections_close_on_initialization_creation_and_replay(self):
        connections = []
        original = Lab.connect

        def track(lab):
            connection = original(lab)
            connections.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as directory, patch.object(Lab, 'connect', track):
            lab = Lab(directory)
            with patch('competition_agent.service.provider.generate_observation', side_effect=OSError('test outage')):
                created = lab.create_demo(request_id='same-request')
            replay = lab.create_demo(request_id='same-request')
            self.assertTrue(replay['idempotent_replay'])
            self.assertEqual(replay['case_id'], created['case_id'])
            self.assertEqual(replay['charged_experiments'], 5)
            for connection in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute('SELECT 1')
            # Windows cannot remove an open SQLite database file.
            (Path(directory)/'cases.sqlite').unlink()

    def test_source_hash_keys_use_portable_separators(self):
        hashes = source_hashes()
        self.assertIn('competition_agent/service.py', hashes)
        self.assertTrue(all('\\' not in key for key in hashes))


if __name__ == '__main__':
    unittest.main()
