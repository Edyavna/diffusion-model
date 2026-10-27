"""Execute the actual notebook in isolated CPU kernels with tiny synthetic data.

No real MNIST, Drive access, browser upload/download, or full training is used.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
notebook = nbformat.read(ROOT / 'MNIST_Diffusion_Colab.ipynb', as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        compile(cell.source, cell.id, 'exec')

synthetic_data = '''
from torch.utils.data import TensorDataset
images = torch.rand(8, 1, 16, 16)
labels = torch.arange(8)
train_loader = DataLoader(TensorDataset(images, labels), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(images, labels), batch_size=BATCH_SIZE)
print('CHECK ONLY: synthetic dataset, no MNIST downloaded')
'''

fake_colab = '''
import types
colab_stub = types.ModuleType('google.colab')
downloaded_files = []
colab_stub.files = types.SimpleNamespace(download=lambda path: downloaded_files.append(path))
sys.modules['google.colab'] = colab_stub
'''

summary = []
for conditional in (True, False):
    with tempfile.TemporaryDirectory(prefix='mnist_colab_check_') as directory:
        work = Path(directory)
        kernels_dir = work / 'kernels'
        kernel_dir = kernels_dir / 'colab-check'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps(dict(
            argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='Isolated notebook CPU check', language='python',
            env={'MPLBACKEND': 'Agg'},
        )), encoding='utf-8')
        candidate = copy.deepcopy(notebook)
        source_by_tag = {}
        for cell in candidate.cells:
            tags = cell.metadata.get('tags', [])
            for tag in tags:
                source_by_tag[tag] = cell.source
            if 'parameters' in tags:
                cell.source = cell.source.replace('CONDITIONAL = True', f'CONDITIONAL = {conditional}')
                cell.source = cell.source.replace('EPOCHS = 6', 'EPOCHS = 1')
                cell.source = cell.source.replace('BATCH_SIZE = 128', 'BATCH_SIZE = 4')
                cell.source = cell.source.replace('PREVIEW_STEPS = 50', 'PREVIEW_STEPS = 2')
                cell.source = cell.source.replace('ALLOW_CPU_TRAINING = False', 'ALLOW_CPU_TRAINING = True')
            elif 'data' in tags:
                cell.source = synthetic_data
            elif 'generate' in tags:
                cell.source = cell.source.replace('COUNT = 40', 'COUNT = 4')
                cell.source = cell.source.replace('GENERATION_STEPS = 50', 'GENERATION_STEPS = 2')
            elif 'download' in tags:
                cell.source = fake_colab + cell.source.replace('DOWNLOAD_RESULTS = False', 'DOWNLOAD_RESULTS = True')

        # Check the archive really contains independently importable model code.
        archive_check = '''
import importlib.util
import zipfile
assert len(downloaded_files) == 1
archive = Path(downloaded_files[0])
with zipfile.ZipFile(archive) as z:
    assert {'best.pth', 'last.pth', 'metrics.csv', 'generated.png',
            'denoising.png', 'model_definition.py', 'vendor/edm_networks.py'}.issubset(z.namelist())
    compile(z.read('model_definition.py'), 'exported_model', 'exec')
spec = importlib.util.spec_from_file_location('exported_model', RUN_DIR / 'model_definition.py')
exported = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exported)
restored = exported.load_checkpoint(RUN_DIR / 'best.pth', 'cpu')
torch.testing.assert_close(
    restored.net.out_conv.weight, model.net.out_conv.weight, rtol=0, atol=0)
trained_checkpoint = RUN_DIR / 'best.pth'
TRAIN_MODEL = False
UPLOAD_CHECKPOINT = False
CHECKPOINT_PATH = str(trained_checkpoint)
RUN_DIR = RUN_DIR.parent / 'generation_only'
'''
        candidate.cells.append(nbformat.v4.new_code_cell(archive_check))
        candidate.cells.append(nbformat.v4.new_code_cell(source_by_tag['load-checkpoint']))
        generation_only = source_by_tag['generate'].replace('COUNT = 40', 'COUNT = 4')
        generation_only = generation_only.replace('GENERATION_STEPS = 50', 'GENERATION_STEPS = 2')
        candidate.cells.append(nbformat.v4.new_code_cell(generation_only))
        candidate.cells.append(nbformat.v4.new_code_cell(
            "assert (RUN_DIR / 'generated.png').is_file()\n"
            "assert model.conditional == " + str(conditional) + "\n"
            "print('PASS: training, export archive, restore, generation-only')"
        ))
        manager = KernelManager(
            kernel_name='colab-check',
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels_dir)]),
        )
        client = NotebookClient(candidate, km=manager, timeout=120,
                                resources={'metadata': {'path': str(work)}})
        try:
            client.execute()
        except Exception:
            nbformat.write(candidate, ROOT / 'colab_check_failed.ipynb')
            raise
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        code_count = sum(cell.cell_type == 'code' for cell in candidate.cells)
        summary.append(dict(conditional=conditional, executed_code_cells=code_count,
                            result='passed', data='synthetic: 8 images',
                            epochs=1, device='cpu'))
        print(f'PASS: conditional={conditional}; {code_count} code cells executed', flush=True)

report = dict(notebook='MNIST_Diffusion_Colab.ipynb', schema='valid',
              syntax='all code cells compiled', checks=summary,
              not_checked=['Cloud Colab runtime', 'CUDA', 'MNIST full training',
                           'real Google Drive mount', 'browser upload/download'])
(ROOT / 'COLAB_CHECK.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('All notebook checks passed. Report: COLAB_CHECK.json')
