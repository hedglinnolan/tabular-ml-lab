"""Turn a failing pytest run's JUnit XML into GitHub annotations, which anyone can read.

CI logs need repository-admin rights to download; check-run annotations do not. So when the
tests fail, this prints a summary and the first failures as ``::error`` lines (GitHub keeps
10 error annotations per step), each with the test id and the end of its message.
"""
import sys
import xml.etree.ElementTree as ET


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def main(path: str) -> None:
    root = ET.parse(path).getroot()
    bad = []
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                test = f"{case.get('classname', '')}::{case.get('name', '')}"
                detail = (node.get("message") or "") + "\n" + (node.text or "")
                bad.append((kind, test, detail.strip()))
    total = sum(int(s.get("tests", 0)) for s in root.iter("testsuite"))
    print(f"::error title=pytest summary::{len(bad)} failing of {total} tests; first: "
          + esc("; ".join(t for _, t, _ in bad[:12])))
    for kind, test, detail in bad[:9]:
        tail = "\n".join(detail.splitlines()[-25:])[-3500:]
        print(f"::error title={kind}: {esc(test)[:180]}::{esc(tail)}")


if __name__ == "__main__":
    main(sys.argv[1])
