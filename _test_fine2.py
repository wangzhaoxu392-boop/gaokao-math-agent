import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import math_agent_all as maa
ok, msg, path = maa.fine_generate_and_save("物理", "牛顿第二定律", "math")
print(ok, "|", msg)
if path:
    import docx
    doc = docx.Document(path)
    n_chars = sum(len(p.text) for p in doc.paragraphs)
    print(f"总字数: {n_chars}")
