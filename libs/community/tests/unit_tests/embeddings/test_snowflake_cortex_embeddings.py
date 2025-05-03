from langchain_community.embeddings import SnowflakeCortexEmbeddings

# ✅ Dummy connection + engine
class DummyConnection:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, query, params):  # Simulate a result from Snowflake
        class DummyResult:
            def fetchone(self):
                # Return a fake embedding (list of floats)
                return [[float(i) for i in range(768)]]
        return DummyResult()

class DummyEngine:
    def connect(self): return DummyConnection()

def test_embed_query(monkeypatch):
    # ✅ Patch create_engine to return our DummyEngine
    from langchain_community.embeddings import snowflake_cortex_engine_embeddings
    monkeypatch.setattr(snowflake_cortex_engine_embeddings, "create_engine", lambda *a, **kw: DummyEngine())

    embedder = SnowflakeCortexEmbeddings(
        model="e5-base-v2",
        embedding_function="EMBED_TEXT_768",
        private_key="fake_private_key_string",
        snowflake_account="fake_account",
        snowflake_username="fake_user",
        snowflake_role="FAKE_ROLE",
        snowflake_database="FAKE_DB",
        snowflake_schema="SCHEMA",
        snowflake_warehouse="CORTEX_WH",
    )

    embedding = embedder.embed_query("LangChain helps orchestrate LLMs.")

    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(val, float) for val in embedding)
