# tests/test_smoke_import.py
def test_package_imports():
    import agentfactory
    assert agentfactory.__version__ == "0.1.0"
