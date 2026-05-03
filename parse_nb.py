import json
import collections
import sys

try:
    with open('ViniCarida_Workshop_Unesp_RioPRetoi.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    print(f'Total cells: {len(cells)}, Code cells: {len(code_cells)}')
    
    imports = collections.Counter()
    def_funcs = []
    classes = []
    
    for c in code_cells:
        src = ''.join(c.get('source', []))
        lines = src.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                parts = line.split(' ')
                if len(parts) >= 2:
                    imports[parts[0] + ' ' + parts[1]] += 1
            if line.startswith('def '):
                def_funcs.append(line.split('(')[0].replace('def ', ''))
            if line.startswith('class '):
                classes.append(line.split('(')[0].split(':')[0].replace('class ', ''))
                
    print('Imports:', list(imports.keys())[:20])
    print('Functions:', def_funcs)
    print('Classes:', classes)
except Exception as e:
    print(f"Error: {e}")
