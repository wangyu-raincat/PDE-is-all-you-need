"""Offline Nexent compatibility contracts; no platform or paid model calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from competition_agent.bundle import build, SCHEMA_REFERENCES
from competition_agent.verification.nexent_v2.schema_reference import ExportAndImportDataFormat

ROOT = Path(__file__).resolve().parents[1]


class NexentV2BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name)

    def test_default_v2_required_fields_and_model_binding(self):
        payload, meta = build(self.dest, 'X', 'http://host.docker.internal:8769/mcp')
        parsed = ExportAndImportDataFormat.model_validate(payload)
        self.assertEqual(meta['nexent_version'], 'v2')
        self.assertEqual(meta['schema_reference_tag'], 'v2.8.5')
        self.assertEqual(meta['schema_reference_commit'], SCHEMA_REFERENCES['v2'][0])
        self.assertFalse(meta['nexent_live_verified'])
        self.assertEqual(len(parsed.agent_info), 6)
        for agent in parsed.agent_info.values():
            self.assertEqual(agent.model_names, ['X'])
            self.assertTrue(agent.enabled)
            self.assertTrue(agent.provide_run_summary)
            self.assertGreater(agent.max_steps, 0)
        self.assertEqual(json.loads((self.dest/'nexent-agents.json').read_text(encoding='utf-8')), payload)

    def test_child_references_form_one_reachable_tree(self):
        payload, _ = build(self.dest)
        agents = payload['agent_info']
        root = payload['agent_id']
        visited = set()
        def visit(agent_id):
            self.assertNotIn(agent_id, visited)
            visited.add(agent_id)
            agent = agents[str(agent_id)]
            self.assertEqual(agent['agent_id'], agent_id)
            self.assertEqual(agent['is_main_agent'], agent_id == root)
            for child in agent['managed_agents']:
                visit(child)
        visit(root)
        self.assertEqual(visited, {int(key) for key in agents})
        self.assertEqual(agents[str(root)]['managed_agents'], [11, 12, 13, 14, 15])

    def test_tool_registration_matches_service_and_catalog(self):
        payload, meta = build(self.dest)
        server = payload['mcp_info'][0]['mcp_server_name']
        names = set()
        for agent in payload['agent_info'].values():
            for tool in agent['tools']:
                self.assertEqual(tool['source'], 'mcp')
                self.assertEqual(tool['usage'], server)
                self.assertEqual(tool['name'], tool['class_name'])
                self.assertEqual(tool['params'], {})
                names.add(tool['class_name'])
        self.assertEqual(server, 'pde_lab_tools')
        self.assertEqual(len(names), meta['mcp_tool_count'])
        self.assertEqual(len(names), 26)
        # Inspect actual registrations, including imported heat tools; no network listener.
        import asyncio
        from competition_agent.mcp_server import build_server
        app = build_server(self.dest / 'mcp-runtime')
        registered = {tool.name for tool in asyncio.run(app.list_tools())}
        self.assertEqual(names, registered)

    def test_six_skill_zips_match_each_agent(self):
        payload, _ = build(self.dest)
        self.assertEqual(len(list(self.dest.glob('*.zip'))), 6)
        for agent in payload['agent_info'].values():
            self.assertEqual(len(agent['skill_names']), 1)
            skill = agent['skill_names'][0]
            with zipfile.ZipFile(self.dest/f'{skill}.zip') as archive:
                self.assertEqual(archive.namelist(), ['SKILL.md'])
                self.assertIsNone(archive.testzip())
                text = archive.read('SKILL.md').decode('utf-8')
            self.assertTrue(text.startswith(f'---\nname: {skill}\ndescription: '))
            self.assertIn('\n---\n', text)

    def test_missing_required_fields_rejected_at_all_levels(self):
        payload, _ = build(self.dest)
        for level, key in [('root', 'agent_info'), ('agent', 'provide_run_summary'),
                           ('tool', 'class_name'), ('mcp', 'mcp_url')]:
            with self.subTest(level=level):
                broken = copy.deepcopy(payload)
                target = {'root': broken, 'agent': broken['agent_info']['11'],
                          'tool': broken['agent_info']['11']['tools'][0],
                          'mcp': broken['mcp_info'][0]}[level]
                del target[key]
                with self.assertRaises(ValueError):
                    ExportAndImportDataFormat.model_validate(broken)

    def test_legacy_preserves_contract_and_original_tools_with_heat_extension(self):
        payload, meta = build(self.dest, nexent_version='legacy')
        expected = json.loads((ROOT/'competition_agent/config/nexent-agents.json').read_text(encoding='utf-8'))
        compatible = copy.deepcopy(payload)
        heat_tools = {'heat_identify', 'heat_holdout_summary', 'heat_compare', 'heat_repeatability'}
        for key, agent in compatible['agent_info'].items():
            self.assertTrue(agent['prompt'])
            # Manufacturing prompts intentionally evolve; all other original fields stay fixed.
            agent['prompt'] = expected['agent_info'][key]['prompt']
            added = {tool['class_name'] for tool in agent['tools']} & heat_tools
            self.assertEqual(added, heat_tools if agent['name'] == 'pde_agent' else set())
            ontology_tools={'ontology_induce','ontology_review_queue','standard_align','kg_query','evidence_retrieve'}
            self.assertEqual({t['class_name'] for t in agent['tools']} & ontology_tools,
                             ontology_tools if agent['name']=='graph_agent' else set())
            agent['tools']=[t for t in agent['tools'] if t['class_name'] not in ontology_tools | {'decision_make','decision_trace_get'}]
            agent['tools'] = [tool for tool in agent['tools'] if tool['class_name'] not in heat_tools]
        self.assertEqual(compatible, expected)
        self.assertEqual(meta['schema_reference_commit'], SCHEMA_REFERENCES['legacy'][0])
        SCHEMA_REFERENCES['legacy'][1].model_validate(payload)

    def test_version_selects_validator_and_unknown_version_has_no_output(self):
        for version in ('legacy', 'v2'):
            with self.subTest(version=version), patch.object(SCHEMA_REFERENCES[version][1], 'model_validate',
                                                          side_effect=ValueError('schema marker')):
                with self.assertRaisesRegex(ValueError, 'schema marker'):
                    build(self.dest/version, nexent_version=version)
        with self.assertRaises(ValueError):
            build(self.dest/'invalid', nexent_version='future')
        self.assertFalse((self.dest/'invalid').exists())

    def test_cli_acceptance_command(self):
        result = subprocess.run([sys.executable, '-m', 'competition_agent.bundle',
                                 '--nexent-version', 'v2', '--model-name', 'X',
                                 '--mcp-url', 'http://host.docker.internal:8769/mcp',
                                 '--output', str(self.dest)], cwd=ROOT, capture_output=True,
                                text=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['nexent_version'], 'v2')
        payload = json.loads((self.dest/'nexent-agents.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['mcp_info'][0]['mcp_url'], 'http://host.docker.internal:8769/mcp')
        ExportAndImportDataFormat.model_validate(payload)


if __name__ == '__main__':
    unittest.main()
