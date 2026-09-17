from speaches.executors.shared import base_model_manager


def test_self_disposing_model_unload_trims_native_heap(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        base_model_manager,
        "_trim_native_heap",
        lambda: calls.append(1),
    )

    holder = base_model_manager.SelfDisposingModel(
        "test/model",
        load_fn=lambda: object(),
        ttl=0,
    )

    with holder:
        pass

    assert holder.model is None
    assert calls == [1]
