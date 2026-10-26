"""Small result-file mirrors. A mounted read is NOT server-side sync proof."""
import os
from pathlib import Path
import shutil


def validate_paths(local, mirror, protected):
    local, mirror = Path(local).resolve(), Path(mirror).resolve()
    if not str(local).startswith('/content/') or '/drive/' in str(local):
        raise ValueError('Primary results must be in local Colab /content, not mounted Drive.')
    for path in [local, *[Path(p).resolve() for p in protected]]:
        if path == mirror or path in mirror.parents or mirror in path.parents:
            raise ValueError('Use a separate NEW mirror, never overwrite source/reference results.')


def copy_file(source, target, immutable):
    from followup_common import sha256
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(str(source)) == sha256(str(target)):
            return
        if immutable:
            raise ValueError('Conflicting saved job/definition; preserve and investigate: '+str(target))
    partial = str(target)+'.partial'
    shutil.copyfile(source, partial)
    if sha256(str(source)) != sha256(partial):
        raise ValueError('Result copy failed; local primary retained.')
    os.replace(partial, target)


def sync_mirror(local, mirror):
    local, mirror = Path(local), Path(mirror)
    for source in local.rglob('*'):
        if source.is_file() and not source.name.endswith('.partial'):
            relative = source.relative_to(local)
            immutable = source.suffix == '.zip' or source.name in ('run_metadata.json','sentence_plan.json')
            copy_file(source, mirror/relative, immutable)


def restore_mirror(local, mirror):
    """Restore before preflight; runner independently verifies every job archive."""
    local, mirror = Path(local), Path(mirror)
    for source in mirror.rglob('*'):
        if source.is_file() and not source.name.endswith('.partial'):
            target = local/source.relative_to(mirror)
            if not target.exists():
                copy_file(source,target,True)
