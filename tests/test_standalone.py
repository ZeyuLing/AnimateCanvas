"""Dependency-closure checks that require no model or GPU."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

class StandaloneTests(unittest.TestCase):
    def test_no_motius_imports(self):
        for file in (ROOT/'animatecanvas').rglob('*.py'):
            tree = ast.parse(file.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                names = ([x.name for x in node.names] if isinstance(node,ast.Import)
                         else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                self.assertFalse(any(n=='motius' or n.startswith('motius.') for n in names),str(file))

    def test_no_framework_install_dependency(self):
        text=(ROOT/'pyproject.toml').read_text()
        dependencies=re.search(r'dependencies\s*=\s*\[(.*?)\]',text,re.S).group(1)
        self.assertNotIn('motius', dependencies.lower())

if __name__=='__main__': unittest.main()
