"""Double-colon model ids must survive provider-prefix normalization."""

from agent.model_metadata import _strip_provider_prefix, get_model_context_length
from providers.base import ProviderProfile


def test_double_colon_model_id_reaches_context_provider_intact(monkeypatch, tmp_path):
    import providers

    model = "tensorzero::function_name::mechanical"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class TensorZeroProfile(ProviderProfile):
        def get_model_context_length(self, model_id):
            assert model_id == model
            return 65536

    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", True)
    providers.register_provider(TensorZeroProfile(name="tensorzero"))

    assert _strip_provider_prefix(model) == model
    assert _strip_provider_prefix("tensorzero:other-model") == "other-model"
    assert get_model_context_length(model, provider="tensorzero") == 65536
