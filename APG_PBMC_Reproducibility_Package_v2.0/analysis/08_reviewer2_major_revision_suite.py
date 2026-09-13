from pathlib import Path

fixed = Path(__file__).with_name('09_reviewer2_major_revision_suite_fixed.py')
code = compile(fixed.read_text(encoding='utf-8'), str(fixed), 'exec')
exec(code, {'__name__': '__main__', '__file__': str(fixed)})
